import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from torch.amp import GradScaler, autocast
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from trainers import vificlip
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
import cv2


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", required=True, type=str)
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--output", type=str, default="exp")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for DistributedDataParallel")
    parser.add_argument("--validate-videos", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="ViFi-CLIP_Few-Shot", help="Name of the MLflow experiment")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the MLflow run")
    args = parser.parse_args()

    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    config = get_config(args)

    return args, config


def main(config, args):
    mlflow.set_tracking_uri("file:/home/ecuser/mlflow_logs")
    mlflow.set_experiment(args.experiment_name)

    if args.run_name is None:
        run_name = "{}_{}_BS{}".format(config.MODEL.ARCH, config.TRAIN.LR, config.TRAIN.BATCH_SIZE)
        if hasattr(config.TRAINER.ViFi_CLIP, 'USE'):
            run_name += "_Freeze-{}".format(config.TRAINER.ViFi_CLIP.USE)
    else:
        run_name = args.run_name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": config.TRAIN.EPOCHS,
            "lr": config.TRAIN.LR,
            "batch_size": config.TRAIN.BATCH_SIZE,
            "weight_decay": config.TRAIN.WEIGHT_DECAY,
        })
        mlflow.log_param("model_arch", config.MODEL.ARCH)
        mlflow.log_param("num_classes", config.DATA.NUM_CLASSES)
        # Rimosso opt_level in quanto AMP è disabilitato
        mlflow.log_param("mixup", config.AUG.MIXUP)
        mlflow.log_param("label_smoothing", config.AUG.LABEL_SMOOTH)
        mlflow.log_param("config_file", os.path.basename(args.config))

        train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
        class_names = [class_name for _, class_name in train_data.classes]

        model = vificlip.returnCLIP(config, logger=logger, class_names=class_names)
        model = model.cuda()

        mixup_fn = None
        if config.AUG.MIXUP > 0:
            criterion = SoftTargetCrossEntropy()
            mixup_fn = CutmixMixupBlending(
                num_classes=config.DATA.NUM_CLASSES,
                smoothing=config.AUG.LABEL_SMOOTH,
                mixup_alpha=config.AUG.MIXUP,
                cutmix_alpha=config.AUG.CUTMIX,
                switch_prob=config.AUG.MIXUP_SWITCH_PROB
            )
        elif config.AUG.LABEL_SMOOTH > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(config, model)
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False
        )

        start_epoch, max_accuracy = 0, 0.0

        if config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(config.OUTPUT)
            if resume_file:
                config.defrost()
                config.MODEL.RESUME = resume_file
                config.freeze()
                logger.info("auto resuming from {}".format(resume_file))
            else:
                logger.info("no checkpoint found in {}".format(config.OUTPUT))

        if config.MODEL.RESUME:
            start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
            if start_epoch > 1:
                logger.info("resetting epochs and max accuracy")
                start_epoch = 0
                max_accuracy = 0
        if config.TEST.ONLY_TEST:
            acc1 = validate(val_loader, model, config)
            logger.info("Test accuracy on {} videos: {:.1f}%".format(len(val_data), acc1))
            return

        for epoch in range(start_epoch, config.TRAIN.EPOCHS):
            train_loader.sampler.set_epoch(epoch)
            train_loss = train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
                acc1, precision, recall, f1, y_true, y_pred = validate(val_loader, model, config, return_preds=True)
                # Logging delle metriche
                mlflow.log_metric("val_acc", acc1, step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_f1", f1, step=epoch)
                # Confusion matrix
                if dist.get_rank() == 0:
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import ConfusionMatrixDisplay
                    import tempfile

                    fig, ax = plt.subplots(figsize=(8, 8))
                    # Calcolo solo le classi presenti in y_true o y_pred
                    present_labels = unique_labels(y_true, y_pred)
                    present_class_names = [class_names[i] for i in present_labels]

                    ConfusionMatrixDisplay.from_predictions(
                        y_true,
                        y_pred,
                        display_labels=present_class_names,
                        ax=ax,
                        xticks_rotation=45,
                        colorbar=False
                    )

                    fig.tight_layout()

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        fig.savefig(tmpfile.name)
                        mlflow.log_artifact(tmpfile.name, artifact_path="confusion_matrices")
                    plt.close(fig)

                logger.info("Val acc: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(acc1, precision, recall, f1))
                is_best = acc1 > max_accuracy
                max_accuracy = max(max_accuracy, acc1)

                if dist.get_rank() == 0 and is_best:
                    epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

        if config.TEST.MULTI_VIEW_INFERENCE:
            config.defrost()
            config.TEST.NUM_CLIP = 4
            config.TEST.NUM_CROP = 3
            config.freeze()
            _, _, _, val_loader = build_dataloader(logger, config)
            acc1 = validate(val_loader, model, config)
            logger.info("Multi-view test acc on {} videos: {:.1f}%".format(len(val_data), acc1))

    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    #scaler = GradScaler("cuda", enabled=(config.TRAIN.OPT_LEVEL != "O0"))

    start = end = time.time()

    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True).reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])
        if mixup_fn:
            images, label_id = mixup_fn(images, label_id)

        output = model(images.float())  # forza float32
        total_loss = criterion(output, label_id) / config.TRAIN.ACCUMULATION_STEPS
        total_loss.backward()

        #with autocast("cuda", enabled=(config.TRAIN.OPT_LEVEL != "O0")):
        #    output = model(images)
        #    total_loss = criterion(output, label_id) / config.TRAIN.ACCUMULATION_STEPS

        #scaler.scale(total_loss).backward()

        if config.TRAIN.ACCUMULATION_STEPS == 1 or (
                config.TRAIN.ACCUMULATION_STEPS > 1 and (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        ):
            #scaler.step(optimizer)
            #scaler.update()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                "Train: [{}/{}][{}/{}]\t" \
                "eta {} lr {:.9f}\t" \
                "time {:.4f} ({:.4f})\t" \
                "loss {:.4f} ({:.4f})\t" \
                "mem {:.0f}MB".format(
                    epoch, config.TRAIN.EPOCHS, idx, num_steps,
                    datetime.timedelta(seconds=int(batch_time.avg * (num_steps - idx))),
                    optimizer.param_groups[0]["lr"],
                    batch_time.val, batch_time.avg,
                    tot_loss_meter.val, tot_loss_meter.avg,
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                )
            )

    logger.info("EPOCH {} training took {}".format(epoch, datetime.timedelta(seconds=int(time.time() - start))))
    return tot_loss_meter.avg


@torch.no_grad()
def validate(val_loader, model, config, return_preds=False):
    model.eval()

    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    y_true, y_pred = [], []

    logger.info("{} views inference".format(config.TEST.NUM_CLIP * config.TEST.NUM_CROP))
    for idx, batch_data in enumerate(val_loader):
        _image = batch_data["imgs"]
        label_id = batch_data["label"].reshape(-1)

        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
        for i in range(n):
            image_input = _image[:, i].cuda(non_blocking=True).float()
            similarity = model(image_input).view(b, -1).softmax(dim=-1)
            tot_similarity += similarity

        preds = tot_similarity.argmax(dim=-1).cpu().numpy()
        targets = label_id.cpu().numpy()
        acc1 = (preds == targets).sum() / b * 100
        acc1_meter.update(acc1, b)
        y_true.extend(targets)
        y_pred.extend(preds)

        if idx % config.PRINT_FREQ == 0:
            logger.info("Test: [{}/{}] Acc@1: {:.3f}".format(idx, len(val_loader), acc1_meter.avg))

    acc1_meter.sync()
    acc1_final = acc1_meter.avg
    if return_preds:
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return acc1_final, precision, recall, f1, y_true, y_pred
    return acc1_final


def validate_videos(dataset_path):
    corrupted_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(root, file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                corrupted_files.append(video_path)
            cap.release()
    return corrupted_files


if __name__ == "__main__":
    # prepare config
    args, config = parse_option()

    if args.validate_videos:
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name="{}".format(config.MODEL.ARCH))
        logger.info("Validating video files in {}".format(config.DATA.ROOT))
        corrupted_files = validate_videos(config.DATA.ROOT)
        if corrupted_files:
            logger.error("Corrupted video files found:")
            for file in corrupted_files:
                logger.error(file)
        else:
            logger.info("No corrupted video files found.")
        exit(0)

    # init_distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print("RANK and WORLD_SIZE in environ: {}/{}".format(rank, world_size))
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name="{}".format(config.MODEL.ARCH))
    logger.info("working dir: {}".format(config.OUTPUT))

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config, args)


