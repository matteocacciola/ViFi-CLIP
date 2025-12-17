from dataclasses import dataclass, field
from logging import Logger
import os
from pathlib import Path
import warnings

import pandas as pd

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.export import Dim
import onnx
from onnxconverter_common.auto_mixed_precision import auto_convert_mixed_precision
from onnxconverter_common.float16 import convert_float_to_float16

import clip


def get_metrics(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MulticlassAccuracy(num_classes, average="macro"),
            "precision": MulticlassPrecision(num_classes, average="macro"),
            "recall": MulticlassRecall(num_classes, average="macro"),
            "f1": MulticlassF1Score(num_classes, average="macro"),
            "accuracy_per_class": MulticlassAccuracy(num_classes, average=None),
            "precision_per_class": MulticlassPrecision(num_classes, average=None),
            "recall_per_class": MulticlassRecall(num_classes, average=None),
            "f1_per_class": MulticlassF1Score(num_classes, average=None),
        }
    )


@dataclass
class ValidationResults:
    """Container for validation metrics results."""

    classes: list[str]

    loss: torch.Tensor

    accuracy: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor

    precision_per_class: torch.Tensor
    recall_per_class: torch.Tensor
    f1_per_class: torch.Tensor
    accuracy_per_class: torch.Tensor

    epoch: int = 0

    def is_better_than(
        self, other: "ValidationResults | None ", metric: str = "loss"
    ) -> bool:
        """Check if current metrics are better than another set."""
        if other is None:
            return True
        elif metric == "loss":
            return self.loss.item() < other.loss.item()
        elif hasattr(self, metric):
            return getattr(self, metric).item() > getattr(other, metric).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def __str__(self) -> str:
        return (
            f"Loss: {self.loss.item():.4f}, Acc: {self.accuracy.item():.3f}, "
            f"P: {self.precision.item():.3f}, R: {self.recall.item():.3f}, F1: {self.f1.item():.3f}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to a DataFrame with per-class rows + macro avg row.

        Returns:
            DataFrame with columns: ['class', 'accuracy', 'precision', 'recall', 'f1']
        """
        rows = []
        # Add per-class rows
        for i, class_name in enumerate(self.classes):
            rows.append(
                {
                    "class": class_name,
                    "accuracy": self.accuracy_per_class[i].item(),
                    "precision": self.precision_per_class[i].item(),
                    "recall": self.recall_per_class[i].item(),
                    "f1": self.f1_per_class[i].item(),
                }
            )

        # Add macro average row
        rows.append(
            {
                "class": "AVG (macro)",
                "accuracy": self.accuracy.item(),
                "precision": self.precision.item(),
                "recall": self.recall.item(),
                "f1": self.f1.item(),
            }
        )

        df = pd.DataFrame(rows)
        df = df.set_index("class")

        return df

    def to_table(self) -> str:
        """Convert metrics to a table string using DataFrame."""
        df = self.to_dataframe()
        return df.to_string(float_format="%.4f")

    def save_to_csv(self, filepath: Path):
        """Save metrics to CSV file using DataFrame."""
        df = self.to_dataframe()
        df.to_csv(filepath, float_format="%.4f")


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def save_checkpoint(
    config,
    epoch,
    model,
    max_accuracy,
    optimizer,
    lr_scheduler,
    save_path: Path,
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }
    torch.save(save_state, save_path)


@torch.inference_mode()
def export_onnx(ddp_model: DDP, working_dir: Path, logger: Logger):
    model = ddp_model.module
    model.eval()
    working_dir = working_dir / "onnx"
    working_dir.mkdir(parents=True, exist_ok=True)
    model_path = working_dir / "model.onnx"

    input = (torch.randn(2, 32, 3, 224, 224, device="cuda"),)

    torch.onnx.export(
        model,
        input,
        model_path,
        input_names=["video"],
        output_names=["logits"],
        dynamic_shapes=[
            (Dim("batch"), Dim("frames"), Dim.STATIC, Dim.STATIC, Dim.STATIC)
        ],
        dynamo=True,
    )
    logger.info("ONNX: exported fp32 version ✅")
    onnx_model = onnx.load(model_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Suppress UserWarnings
        # model_fp16 = auto_convert_mixed_precision(
        #     onnx_model,
        #     {"video": input[0].numpy(force=True)},
        #     rtol=0.01,
        #     atol=0.001,
        #     keep_io_types=True,
        # )
        model_fp16 = convert_float_to_float16(
            onnx_model,
            keep_io_types=True,
        )

    model_fp16_path = model_path.with_stem("model_fp16")
    onnx.save(
        model_fp16,
        model_fp16_path,
        save_as_external_data=True,
        location=f"{model_fp16_path.name}.data",
    )
    logger.info("ONNX: exported fp16 version ✅")
    logger.info(f"ONNX models saved in {model_path.parent}")


def load_model_checkpoint(model: nn.Module, checkpoint, logger):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Resume model: {msg}")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME):
        logger.info(
            f"==============> Resuming form {config.MODEL.RESUME}...................."
        )
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
        load_state_dict = checkpoint["model"]

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            start_epoch = checkpoint["epoch"] + 1
            max_accuracy = checkpoint["max_accuracy"]

            logger.info(
                f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
            )

            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.0

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat(
        [clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes]
    )

    return classes
