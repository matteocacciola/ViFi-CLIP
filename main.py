import os
from typing import Tuple, Dict
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
import argparse
import datetime
import shutil
import time
import numpy as np
import random

import cv2
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from yacs.config import CfgNode

from datasets.build import build_dataloader
from datasets.blending import CutmixMixupBlending
from trainers import vificlip
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import (
    AverageMeter,
    epoch_saving,
    load_checkpoint,
    auto_resume_helper,
    load_model_checkpoint,
    model_onnx_conversion,
)
from utils.logger import create_logger
from utils.config import get_config


@dataclass
class TrainingMetrics:
    accuracy: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0
    loss: float = float("inf")
    y_true: list[int] = field(default_factory=lambda: [])
    y_pred: list[int] = field(default_factory=lambda: [])


class ViFiCLIPTrainer:
    def __init__(self, args, config: CfgNode):
        """Initialize trainer with configuration and command line arguments."""
        self.args = args
        self.config = config
        self._validate_inputs()

        # Training state
        self.current_epoch = 0
        self.best_metrics = TrainingMetrics()

        # Initialize components
        self._init_distributed()
        self._set_random_seeds()
        self._init_logging()
        self._init_training_components()

    def _validate_inputs(self):
        """Validate configuration and arguments."""
        if not Path(self.args.config).exists():
            raise FileNotFoundError(f"Config file not found: {self.args.config}")
        if self.config.DATA.NUM_CLASSES <= 1:
            raise ValueError("Number of classes must be greater than 1")

    def _init_distributed(self):
        """Initialize distributed training environment."""
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices available")

        self.rank = int(os.environ.get("RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.cuda.set_device(self.args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        dist.barrier(device_ids=[self.args.local_rank])

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = True

    def _init_logging(self):
        """Initialize logging system."""
        self.log_dir = Path(self.config.OUTPUT)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(
            output_dir=self.log_dir,
            dist_rank=dist.get_rank(),
            name=self.config.MODEL.ARCH,
        )
        self._log_configuration()

    def _log_configuration(self):
        """Log the training configuration."""
        self.logger.info(f"Logs dir: {self.log_dir}")
        self.logger.info(f"Training configuration:\n{self.config}")
        if dist.get_rank() == 0:
            config_copy_path = self.log_dir / "config.yaml"
            shutil.copy(self.args.config, config_copy_path)
            self.logger.info(f"Config file saved to {config_copy_path}")

    def _init_training_components(self):
        """Initialize all components needed for training."""
        self._init_data_loaders()
        self._init_model()
        self._init_loss_function()
        self._init_optimizer()
        self._load_checkpoints()

    def _init_data_loaders(self):
        """Initialize data loaders for training and validation."""
        try:
            self.train_data, self.val_data, self.train_loader, self.val_loader = (
                build_dataloader(self.logger, self.config)
            )
            self.class_names = [class_name for _, class_name in self.train_data.classes]
        except Exception as e:
            self.logger.error(f"Failed to initialize data loaders: {str(e)}")
            raise

    def _init_model(self):
        """Initialize the model and move to GPU."""
        try:
            model = vificlip.returnCLIP(
                self.config, logger=self.logger, class_names=self.class_names
            ).cuda()

            self.model = DDP(
                model,
                device_ids=[self.config.LOCAL_RANK],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _init_loss_function(self):
        """Initialize loss function and mixup augmentation."""
        if self.config.AUG.MIXUP > 0:
            self.criterion = SoftTargetCrossEntropy()
            self.mixup_fn = CutmixMixupBlending(
                num_classes=self.config.DATA.NUM_CLASSES,
                smoothing=self.config.AUG.LABEL_SMOOTH,
                mixup_alpha=self.config.AUG.MIXUP,
                cutmix_alpha=self.config.AUG.CUTMIX,
                switch_prob=self.config.AUG.MIXUP_SWITCH_PROB,
            )
        elif self.config.AUG.LABEL_SMOOTH > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config.AUG.LABEL_SMOOTH
            )
            self.mixup_fn = None
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.mixup_fn = None

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = build_optimizer(self.config, self.model)
        self.lr_scheduler = build_scheduler(
            self.config, self.optimizer, len(self.train_loader)
        )
        self.scaler = GradScaler()

    def _load_checkpoints(self):
        """Handle checkpoint loading and auto-resume functionality."""
        if self.config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(self.config.OUTPUT)
            if resume_file:
                self.config.defrost()
                self.config.MODEL.RESUME = resume_file
                self.config.freeze()
                self.logger.info(f"Auto resuming from {resume_file}")

        if self.config.MODEL.RESUME:
            self.current_epoch, self.best_metrics.accuracy = load_checkpoint(
                self.config, self.model, self.optimizer, self.lr_scheduler, self.logger
            )
            if self.current_epoch > 1:
                self.logger.info(
                    "Resetting epoch counter & best metrics after loading weights"
                )
                self.current_epoch = 0
                self.best_metrics = TrainingMetrics()

    @staticmethod
    def _if_mlflow(alt=lambda: None):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if self.args.mlflow:
                    return func(self, *args, **kwargs)
                else:
                    return alt()

            return wrapper

        return decorator

    def run(self):
        """Main training loop entry point."""
        if self.config.TEST.ONLY_TEST:
            test_metrics = self.validate()
            self.logger.info(
                f"Test accuracy: {test_metrics.accuracy:.2f}% "
                f"on {len(self.val_data)} videos"
            )
            return

        with self._init_mlflow_run():
            try:
                for epoch in range(self.current_epoch, self.config.TRAIN.EPOCHS):
                    train_loss = self._train_epoch(epoch)
                    self._log_mlflow_metrics(epoch, "train_loss", train_loss)

                    if self._should_validate(epoch):
                        val_metrics = self.validate(return_predictions=True)
                        self._log_mlflow_validation(epoch, val_metrics)
                        self._process_validation_results(epoch, val_metrics)

                    self._optional_multiview_inference()

            finally:
                self._save_final_model()
                if dist.is_initialized():
                    dist.destroy_process_group()

    @_if_mlflow(alt=lambda: nullcontext())
    def _init_mlflow_run(self):
        """Initialize MLflow run context."""
        mlflow.set_tracking_uri("file:./.mlflow_logs")
        mlflow.set_experiment(self.args.experiment_name)
        
        if self.args.run_name:
            run_name = self.args.run_name
        else:
            name_parts = [
                self.config.MODEL.ARCH,
                f"LR-{self.config.TRAIN.LR}",
                f"BS-{self.config.TRAIN.BATCH_SIZE}",
            ]

            if hasattr(self.config.TRAINER.ViFi_CLIP, "USE"):
                name_parts.append(f"Freeze-{self.config.TRAINER.ViFi_CLIP.USE}")

            run_name = "_".join(name_parts)
        run = mlflow.start_run(run_name=run_name)

        mlflow.log_params(
            {
                "epochs": self.config.TRAIN.EPOCHS,
                "lr": self.config.TRAIN.LR,
                "batch_size": self.config.TRAIN.BATCH_SIZE,
                "weight_decay": self.config.TRAIN.WEIGHT_DECAY,
                "model_arch": self.config.MODEL.ARCH,
                "num_classes": self.config.DATA.NUM_CLASSES,
                "mixup": self.config.AUG.MIXUP,
                "label_smoothing": self.config.AUG.LABEL_SMOOTH,
                "config_file": os.path.basename(self.args.config),
            }
        )

        return run

    def _train_epoch(self, epoch: int) -> float:
        """Train model for one epoch."""
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)
        self.optimizer.zero_grad()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        start = time.time()
        batch_start = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Process batch
            images, labels = self._prepare_batch(batch_data)
            loss = self._train_batch(images, labels, epoch, batch_idx)

            # Update metrics
            loss_meter.update(loss.item(), len(labels))
            batch_time.update(time.time() - batch_start)
            batch_start = time.time()

            # Log progress
            if batch_idx % self.config.PRINT_FREQ == 0:
                self._log_batch_progress(
                    epoch, batch_idx, len(self.train_loader), batch_time, loss_meter
                )

        self._log_epoch_completion(epoch, loss_meter.avg, start)
        return loss_meter.avg

    def _prepare_batch(self, batch_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training."""
        images = batch_data["imgs"].cuda(non_blocking=True)
        labels = batch_data["label"].cuda(non_blocking=True).reshape(-1)
        images = images.view((-1, self.config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if self.mixup_fn:
            images, labels = self.mixup_fn(images, labels)

        return images, labels

    def _train_batch(
        self, images: torch.Tensor, labels: torch.Tensor, epoch: int, batch_idx: int
    ) -> torch.Tensor:
        """Process a single training batch."""

        with autocast(device_type="cuda"):
            outputs = self.model(images.float())
            loss = (
                self.criterion(outputs, labels) / self.config.TRAIN.ACCUMULATION_STEPS
            )

        self.scaler.scale(loss).backward()

        if (batch_idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0 or (
            batch_idx + 1 == len(self.train_loader)
        ):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step_update(epoch * len(self.train_loader) + batch_idx)

        return loss

    def _log_batch_progress(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        batch_time: AverageMeter,
        loss_meter: AverageMeter,
    ):
        """Log training progress for current batch."""
        remaining_time = batch_time.avg * (total_batches - batch_idx)
        self.logger.info(
            f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{batch_idx}/{total_batches}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"ETA {datetime.timedelta(seconds=int(remaining_time))}\t"
            f"LR {self.optimizer.param_groups[0]['lr']:.6f}\t"
            f"Mem {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB"
        )

    def _log_epoch_completion(self, epoch: int, avg_loss: float, start_time: float):
        """Log completion of training epoch."""
        self.logger.info(f"Train [{datetime.timedelta(seconds=int(time.time() - start_time))}] - Epoch {epoch}\t Loss: {avg_loss:.4f}")

    @torch.inference_mode()
    def validate(self, val_loader=None, return_predictions=False) -> TrainingMetrics:
        """Validate model performance."""
        val_loader = val_loader or self.val_loader
        self.model.eval()

        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        metrics = TrainingMetrics()
        self.logger.info(
            f"Running validation with {self.config.TEST.NUM_CLIP * self.config.TEST.NUM_CROP} views"
        )

        for batch_idx, batch_data in enumerate(val_loader):
            images, labels = self._prepare_validation_batch(batch_data)
            outputs = self._process_validation_batch(images)

            # Update metrics
            loss = F.cross_entropy(outputs, labels).item()
            preds = outputs.argmax(dim=-1).numpy(force=True)
            targets = labels.numpy(force=True)
            accuracy = (preds == targets).mean() * 100

            acc_meter.update(accuracy, len(images))
            loss_meter.update(loss, len(images))
            if return_predictions:
                metrics.y_true.extend(targets)
                metrics.y_pred.extend(preds)

            # Log progress
            if batch_idx % self.config.PRINT_FREQ == 0:
                self.logger.info(
                    f"Validation: [{batch_idx}/{len(val_loader)}] "
                    f"Acc@1: {acc_meter.avg:.3f}"
                )

        # Finalize metrics
        acc_meter.sync()
        loss_meter.sync()
        metrics.accuracy = acc_meter.avg
        metrics.loss = loss_meter.avg

        if return_predictions:
            metrics.precision = precision_score(
                metrics.y_true, metrics.y_pred, average="macro", zero_division=0
            ) # type: ignore
            metrics.recall = recall_score(
                metrics.y_true, metrics.y_pred, average="macro", zero_division=0
            ) # type: ignore
            metrics.f1_score = f1_score(
                metrics.y_true, metrics.y_pred, average="macro", zero_division=0
            ) # type: ignore

        return metrics

    def _prepare_validation_batch(
        self, batch_data: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for validation."""
        images = batch_data["imgs"].cuda(non_blocking=True)
        labels = batch_data["label"].reshape(-1).cuda(non_blocking=True)
        return images, labels

    def _process_validation_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Process validation batch through model."""
        b, tn, c, h, w = images.size()
        t = self.config.DATA.NUM_FRAMES
        n = tn // t
        images = images.view(b, n, t, c, h, w)

        outputs = torch.zeros((b, self.config.DATA.NUM_CLASSES)).cuda()
        for i in range(n):
            output = self.model(images[:, i].float()).view(b, -1).softmax(dim=-1)
            outputs += output

        return outputs

    def _should_validate(self, epoch: int) -> bool:
        """Determine if validation should run this epoch."""
        return epoch % self.config.SAVE_FREQ == 0 or epoch == (
            self.config.TRAIN.EPOCHS - 1
        )

    def _process_validation_results(self, epoch: int, metrics: TrainingMetrics):
        """Process and log validation results."""
        self.logger.info(
            f"Validation - Epoch: {epoch}\n"
            f"Accuracy: {metrics.accuracy:.2f}%\t"
            f"Precision: {metrics.precision:.2f}\t"
            f"Recall: {metrics.recall:.2f}\t"
            f"F1: {metrics.f1_score:.2f}\t"
            f"Loss: {metrics.loss:.6f}"
        )

        is_best = metrics.loss < self.best_metrics.loss
        if is_best:
            self.best_metrics = metrics

        if dist.get_rank() == 0 and is_best:
            epoch_saving(
                self.config,
                epoch,
                self.model,
                self.best_metrics.accuracy,
                self.optimizer,
                self.lr_scheduler,
                self.logger,
                self.config.OUTPUT,
                is_best,
            )
        dist.barrier(device_ids=[self.args.local_rank])

    @_if_mlflow()
    def _log_mlflow_validation(self, epoch: int, metrics: TrainingMetrics):
        """Log validation metrics to MLflow."""
        mlflow.log_metrics(
            {
                "val_acc": metrics.accuracy,
                "val_precision": metrics.precision,
                "val_recall": metrics.recall,
                "val_f1": metrics.f1_score,
                "val_loss": metrics.loss,
            },
            step=epoch,
        )

        if dist.get_rank() == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            present_labels = unique_labels(metrics.y_true, metrics.y_pred)
            present_class_names = [self.class_names[i] for i in present_labels]

            ConfusionMatrixDisplay.from_predictions(
                metrics.y_true,
                metrics.y_pred,
                display_labels=present_class_names,
                ax=ax,
                xticks_rotation=45,
                colorbar=False,
            )

            fig.tight_layout()

            fig_path = f"confusion_matrices/{epoch}.png"
            mlflow.log_figure(fig, fig_path)
            plt.close(fig)

    def _optional_multiview_inference(self):
        """Run multi-view inference if configured."""
        if self.config.TEST.MULTI_VIEW_INFERENCE:
            self.config.defrost()
            self.config.TEST.NUM_CLIP = 4
            self.config.TEST.NUM_CROP = 3
            self.config.freeze()

            _, val_data, _, val_loader = build_dataloader(self.logger, self.config)
            metrics = self.validate(val_loader)

            self.logger.info(
                f"Multi-view test accuracy: {metrics.accuracy:.1f}% "
                f"on {len(val_data)} videos"
            )

    def _save_final_model(self):
        """Save final model and convert to ONNX."""
        if dist.get_rank() == 0:
            best_model_path = self.log_dir / "best.pth"
            load_model_checkpoint(self.model, best_model_path, self.logger)
            model_onnx_conversion(self.model, self.log_dir, self.logger)

    @_if_mlflow()
    def _log_mlflow_metrics(self, epoch: int, metric_name: str, value: float):
        """Log training metrics to MLflow."""
        mlflow.log_metric(metric_name, value, step=epoch)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", required=True, type=str)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument("--validate-videos", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ViFi-CLIP_Few-Shot",
        help="Name of the MLflow experiment",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Name of the MLflow run"
    )
    args = parser.parse_args()

    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    config = get_config(args)

    return args, config


def validate_videos(dataset_path):
    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}"
    )
    logger.info(f"Validating video files in {config.DATA.ROOT}")
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
    if corrupted_files:
        logger.error("Corrupted video files found:")
        for file in corrupted_files:
            logger.error(file)
    else:
        logger.info("No corrupted video files found.")


if __name__ == "__main__":
    args, config = parse_option()

    if args.validate_videos:
        validate_videos(config.DATA.ROOT)
    else:
        trainer = ViFiCLIPTrainer(args, config)
        trainer.run()
