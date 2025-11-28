import os
import argparse
import shutil
import time
from pathlib import Path
from typing import Optional
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
from yacs.config import CfgNode

# Project imports
from datasets.build import build_dataloader
from datasets.blending import CutmixMixupBlending
from trainers import vificlip
from utils.optimizer import build_optimizer, build_scheduler
from torcheval.metrics import Mean
from torcheval.metrics.toolkit import sync_and_compute
from utils.tools import (
    TimeMeter,
    ValidationMeter,
    ValidationMetrics,
    load_model_checkpoint,
    export_onnx,
    save_checkpoint,
    load_checkpoint,
    auto_resume_helper,
)
from utils.logger import create_logger
from utils.config import get_config


BEST_CHECKPOINT_NAME = "best.pth"


class ViFiCLIPTrainer:
    """Simplified trainer with improved progress tracking and checkpoint management."""

    def __init__(self, config: CfgNode, args: argparse.Namespace):
        """Initialize trainer with config and arguments."""
        self._init_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.config = config
        self.args = args
        # Setup output directory and logger
        self.output_dir = Path(config.OUTPUT)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_logger(
            output_dir=self.output_dir, dist_rank=self.rank, name=config.MODEL.ARCH
        )

        # Set random seeds for reproducibility
        seed = config.SEED + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = True

        # Log and save configuration
        self.logger.info(f"Configuration:\n{config}")
        self.logger.info(f"Distributed setup: rank={self.rank}/{self.world_size}")

        if self.is_main_process:
            config_file = self.output_dir / "config.yaml"
            shutil.copy(args.config, config_file)
            self.logger.info(f"Config saved to {config_file}")

        # Initialize all training components
        self._init_components()

    def _init_distributed(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.is_main_process = self.rank == 0
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.local_rank)
        dist.barrier(device_ids=[self.local_rank])

    def _init_components(self):
        """Initialize data loaders, model, optimizer, etc."""
        # Load data
        self.logger.info("Loading datasets...")
        self.train_data, self.val_data, self.train_loader, self.val_loader = (
            build_dataloader(self.config)
        )
        self.class_names = self.train_data.classes
        self.num_classes = len(self.class_names)
        self.logger.info(
            f"Loaded {len(self.train_data)} train, {len(self.val_data)} val samples"
        )

        # Build model
        model = vificlip.returnCLIP(
            self.config, logger=self.logger, class_names=self.class_names
        ).to(self.device)

        # Wrap model for distributed training if needed
        self.model = DDP(model, device_ids=[self.local_rank])

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            f"Model has {total_params / 1e6:.2f}M total params, "
            f"{trainable_params / 1e6:.2f}M trainable"
        )

        # Setup loss functions with optional class weighting
        class_weight = None
        if hasattr(self.train_data, "class_probs"):
            class_weight = 1.0 / self.train_data.class_probs
            class_weight = (class_weight / class_weight.sum()).to(self.device)
            self.logger.info(f"Using balanced class weights: {class_weight}")

        # Validation loss always uses class weights if available
        self.val_criterion = nn.CrossEntropyLoss(weight=class_weight)

        # Training loss with optional mixup/cutmix
        if self.config.AUG.MIXUP > 0 or self.config.AUG.CUTMIX > 0:
            self.logger.info(
                f"Using Mixup({self.config.AUG.MIXUP}) + "
                f"Cutmix({self.config.AUG.CUTMIX})"
            )
            self.train_criterion = nn.CrossEntropyLoss(weight=class_weight)
            self.mixup_fn = CutmixMixupBlending(
                num_classes=self.num_classes,
                smoothing=self.config.AUG.LABEL_SMOOTH,
                mixup_alpha=self.config.AUG.MIXUP,
                cutmix_alpha=self.config.AUG.CUTMIX,
                switch_prob=self.config.AUG.MIXUP_SWITCH_PROB,
            )
        else:
            self.train_criterion = nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=self.config.AUG.LABEL_SMOOTH,
            )
            self.mixup_fn = None

        # Optimizer and scheduler
        self.optimizer = build_optimizer(self.config, self.model)
        self.lr_scheduler = build_scheduler(
            self.config, self.optimizer, len(self.train_loader)
        )

        # Mixed precision scaler
        self.scaler = GradScaler()

        # Training state
        self.start_epoch = 0
        self.best_metrics = ValidationMetrics()
        self.best_metric_name = self.config.TRAIN.BEST_METRIC

        # Handle checkpoint loading/resuming
        self._resume_checkpoint()
        dist.barrier(device_ids=[self.local_rank])

        # Setup MLflow if enabled
        self._setup_mlflow()

    def _resume_checkpoint(self):
        """Load checkpoint with auto-resume support."""
        # Auto-resume: find latest checkpoint
        if self.config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(self.output_dir)
            if resume_file:
                self.config.defrost()
                self.config.MODEL.RESUME = resume_file
                self.config.freeze()
                self.logger.info(f"Auto-resuming from {resume_file}")

        # Load checkpoint if specified
        if self.config.MODEL.RESUME:
            self.start_epoch, best_acc = load_checkpoint(
                self.config, self.model, self.optimizer, self.lr_scheduler, self.logger
            )

            # If loading pretrained weights (not resuming), reset epoch
            if self.config.MODEL.PRETRAINED and self.start_epoch > 0:
                self.logger.info(
                    "Loading pretrained weights only, resetting epoch counter"
                )
                self.start_epoch = 0
            elif best_acc > 0:
                self.best_metrics.accuracy = best_acc
                self.logger.info(
                    f"Resumed from epoch {self.start_epoch} with best acc {best_acc:.2f}%"
                )

    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        self.mlflow_enabled = self.args.mlflow and self.is_main_process

        if self.mlflow_enabled:
            try:
                mlflow.set_tracking_uri("file:./.mlflow_logs")
                mlflow.set_experiment(self.args.experiment_name)

                # Generate run name
                run_name = self.args.run_name or (
                    f"{self.config.MODEL.ARCH}_"
                    f"lr{self.config.TRAIN.LR}_"
                    f"bs{self.config.TRAIN.BATCH_SIZE}"
                )

                mlflow.start_run(run_name=run_name)

                # Log hyperparameters
                mlflow.log_params(
                    {
                        "model": self.config.MODEL.ARCH,
                        "epochs": self.config.TRAIN.EPOCHS,
                        "batch_size": self.config.TRAIN.BATCH_SIZE,
                        "learning_rate": self.config.TRAIN.LR,
                        "num_classes": self.num_classes,
                        "num_frames": self.config.DATA.NUM_FRAMES,
                        "mixup": self.config.AUG.MIXUP,
                        "cutmix": self.config.AUG.CUTMIX,
                    }
                )

                self.logger.info(f"MLflow tracking enabled: {run_name}")

            except Exception as e:
                self.logger.warning(f"Failed to setup MLflow: {e}")
                self.mlflow_enabled = False

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with detailed progress tracking."""
        self.model.train()

        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        # Metrics
        loss_meter = Mean()
        time_meter = TimeMeter()

        # Progress bar (only show on main process)
        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}/{self.config.TRAIN.EPOCHS}",
            disable=not self.is_main_process,
        )

        end = time.time()

        for batch_idx, batch_data in enumerate(pbar):
            # Measure data loading time
            time_meter.data_meter.update(torch.tensor(time.time() - end))

            # Prepare batch
            images = batch_data["imgs"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True).flatten()
            # Reshape for video frames [B, T, C, H, W]
            if images.size(1) != self.config.DATA.NUM_FRAMES:
                images = images.view(-1, self.config.DATA.NUM_FRAMES, *images.shape[2:])

            # Apply mixup/cutmix if enabled
            if self.mixup_fn:
                images, labels = self.mixup_fn(images, labels)

            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                outputs = self.model(images)
                loss = self.train_criterion(outputs, labels)
                loss_meter.update(loss.cpu(), weight=len(outputs))
                loss = loss / self.config.TRAIN.ACCUMULATION_STEPS
            # Backward pass
            self.scaler.scale(loss).backward()

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0 or (
                batch_idx + 1
            ) == len(self.train_loader):
                # Optional gradient clipping for stability
                if self.config.TRAIN.get("CLIP_GRAD", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.TRAIN.CLIP_GRAD
                    )
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # Update learning rate
                if self.lr_scheduler:
                    self.lr_scheduler.step_update(
                        epoch * len(self.train_loader) + batch_idx
                    )

            # Update metrics
            time_meter.batch_meter.update(torch.tensor(time.time() - end))
            # Update progress bar with timing info
            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                loss_avg = sync_and_compute(loss_meter).item()
            else:
                loss_avg = loss_meter.compute().item()
            data_avg = time_meter.data_time
            batch_avg = time_meter.batch_time
            if self.is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_avg:.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        "data_t": f"{data_avg:.2f}s",
                        "batch_t": f"{batch_avg:.2f}s",
                    }
                )

        loss_meter.reset()
        time_meter.reset()

        return loss_avg

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> ValidationMetrics:
        """
        Validate the model with proper distributed metrics handling.

        Note on distributed metrics:
        - loss and accuracy are synchronized across GPUs using AverageMeter.sync()
        - precision/recall/f1 are only computed on main process from gathered predictions
        - This ensures consistency and correctness in distributed settings
        """
        self.model.eval()
        loader = loader or self.val_loader

        # Metrics using AverageMeter for proper distributed sync
        val_meter = ValidationMeter(classes=self.class_names)
        time_meter = TimeMeter()

        # Collect predictions only on main process to save memory

        pbar = tqdm(loader, desc="Validation", disable=not self.is_main_process)

        end = time.time()
        for batch_data in pbar:
            # Measure data loading time
            time_meter.data_meter.update(torch.tensor(time.time() - end))

            # Prepare batch
            images = batch_data["imgs"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True).flatten()

            # Multi-view inference: average predictions across views
            b, tn, c, h, w = images.size()
            t = self.config.DATA.NUM_FRAMES
            n = tn // t  # Number of views
            images = images.view(b * n, t, c, h, w)
            labels = labels.repeat_interleave(n)

            with autocast(device_type="cuda"):
                logits = self.model(images)
                loss = self.val_criterion(logits, labels)
                val_meter.update_loss(loss, n_samples=len(labels))

            logits = logits.view(b, n, -1).mean(dim=1)

            # Compute metrics
            preds = logits.argmax(dim=-1)
            val_meter.update(preds, labels)

            time_meter.batch_meter.update(torch.tensor(time.time() - end))
            end = time.time()

            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{val_meter.loss:.4f}",
                        "acc": f"{val_meter.accuracy:.3f}",
                        "data_t": f"{time_meter.data_time:.3f}s",
                        "batch_t": f"{time_meter.batch_time:.3f}s",
                    }
                )

        # Synchronize and compute metrics across all GPUs
        metrics = val_meter.get_val_metrics()

        val_meter.reset()
        return metrics

    def _save_checkpoint(
        self, epoch: int, metrics: ValidationMetrics, force_save: bool = False
    ):
        """
        Smart checkpoint saving strategy:
        - Always track and save the best model
        - Save regular checkpoints based on SAVE_FREQ
        - Avoid saving unnecessary intermediate checkpoints
        """
        # Check if this is the best model so far
        is_best = metrics.is_better_than(self.best_metrics, self.best_metric_name)
        if is_best:
            self.best_metrics = metrics
            self.logger.info(
                f"New best model! {self.best_metric_name}: "
                f"{getattr(metrics, self.best_metric_name):.4f}"
            )

        # Determine if we should save a regular checkpoint
        is_intermediat_save = force_save or epoch % self.config.SAVE_FREQ == 0

        # Save checkpoint (main process only)
        save_path = self.output_dir / "last.pth"
        save_checkpoint(
            self.config,
            epoch,
            self.model,
            metrics.accuracy,
            self.optimizer,
            self.lr_scheduler,
            save_path=save_path,
        )
        if is_best:
            best_path = self.output_dir / BEST_CHECKPOINT_NAME
            shutil.copy(save_path, best_path)
        if is_intermediat_save:
            intermediate_path = self.output_dir / f"epoch_{epoch}.pth"
            shutil.copy(save_path, intermediate_path)

    def train(self):
        """Main training loop with improved checkpoint management."""
        # Test-only mode
        if self.config.TEST.ONLY_TEST:
            self.logger.info("Running test-only evaluation...")
            metrics = self.validate()
            self.logger.info(f"Test Results: {metrics}")
            return

        self.logger.info(f"Starting training from epoch {self.start_epoch}")
        self.logger.info(f"Best metric to track: {self.best_metric_name}")
        self.logger.info(
            f"Checkpoint save frequency: every {self.config.SAVE_FREQ} epochs"
        )

        # Early stopping setup
        patience = self.config.TRAIN.get("EARLY_STOPPING_PATIENCE", 0)
        patience_counter = 0
        best_early_stop_value = float("inf") if self.best_metric_name == "loss" else 0

        for epoch in range(self.start_epoch, self.config.TRAIN.EPOCHS):
            epoch_start = time.time()

            # Training phase
            train_loss = self.train_epoch(epoch)

            # Log training metrics
            if self.mlflow_enabled:
                mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Validation phase (always validate to track best model)
            val_metrics = self.validate()

            # Log validation results
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            self.logger.info(f"Epoch {epoch} - Validation Loss: {val_metrics.loss:.4f}\n{val_metrics.to_table()}")

            # MLflow logging
            if self.mlflow_enabled:
                mlflow.log_metrics(
                    {
                        "val_loss": val_metrics.loss,
                        "val_accuracy": val_metrics.accuracy,
                        "val_precision": val_metrics.precision,
                        "val_recall": val_metrics.recall,
                        "val_f1": val_metrics.f1,
                    },
                    step=epoch,
                )

            # Save checkpoint (handles best model tracking internally)
            if self.is_main_process:
                self._save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if patience > 0:
                current_value = getattr(val_metrics, self.best_metric_name)

                # Check if improved
                if self.best_metric_name == "loss":
                    improved = current_value < best_early_stop_value
                else:
                    improved = current_value > best_early_stop_value

                if improved:
                    best_early_stop_value = current_value
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self.logger.info(
                        f"No improvement for {patience_counter}/{patience} epochs"
                    )

                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

            # Log epoch time
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch} completed in "
                f"{datetime.timedelta(seconds=int(epoch_time))}\n"
                + "=" * 50
            )

        # Save metrics
        if self.is_main_process:
            metrics_path = self.output_dir / "best_metrics.csv"
            self.best_metrics.save_to_csv(metrics_path)

        # Save the onnx conversion of the best model
        best_path = self.output_dir / BEST_CHECKPOINT_NAME
        self.logger.info("Loading the best model from: %s", best_path)
        load_model_checkpoint(self.model, best_path, self.logger)
        if self.is_main_process:
            export_onnx(self.model, self.output_dir, self.logger)

        # Multi-view test if configured
        if self.config.TEST.MULTI_VIEW_INFERENCE:
            self._run_multiview_test()

        # Final summary
        self.logger.info(
            f"Training complete! Loading model with best {self.best_metric_name}.\n" 
            f"Loss: {self.best_metrics.loss:.4f}\n\n"
            f"{self.best_metrics.to_table()}"
        )

    def _run_multiview_test(self):
        """Run enhanced multi-view testing."""
        self.logger.info("Running multi-view inference test...")

        # Temporarily update config for more views
        original_clip = self.config.TEST.NUM_CLIP
        original_crop = self.config.TEST.NUM_CROP

        self.config.defrost()
        self.config.TEST.NUM_CLIP = 4
        self.config.TEST.NUM_CROP = 3
        self.config.freeze()

        # Build new test loader with more views
        _, _, _, test_loader = build_dataloader(self.config)

        # Run validation
        test_metrics = self.validate(test_loader)

        self.logger.info(
            f"Multi-view Test ({self.config.TEST.NUM_CLIP}x{self.config.TEST.NUM_CROP} views): "
            f"{test_metrics}"
        )

        # Restore original config
        self.config.defrost()
        self.config.TEST.NUM_CLIP = original_clip
        self.config.TEST.NUM_CROP = original_crop
        self.config.freeze()

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.mlflow_enabled:
            mlflow.end_run()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ViFi-CLIP Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--config", "-cfg", required=True, help="Path to config file")

    # Optional overrides
    parser.add_argument("--opts", nargs="+", help="Modify config using KEY VALUE pairs")
    parser.add_argument("--output", type=str, help="Output directory override")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--pretrained", type=str, help="Load pretrained weights")
    parser.add_argument("--only-test", action="store_true", help="Test only mode")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument(
        "--accumulation-steps", type=int, help="Gradient accumulation steps"
    )

    # Distributed
    parser.add_argument(
        "--local-rank", type=int, default=-1, help="Local rank for distributed training"
    )

    # MLflow
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ViFi-CLIP",
        help="MLflow experiment name",
    )
    parser.add_argument("--run-name", type=str, help="MLflow run name")

    args = parser.parse_args()

    # Get local rank from environment if not specified
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = get_config(args)

    # Basic validation
    if not Path(config.DATA.ROOT).exists():
        raise ValueError(f"Data root not found: {config.DATA.ROOT}")

    # Create trainer
    trainer = ViFiCLIPTrainer(config, args)

    try:
        # Run training
        trainer.train()
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
    except Exception as e:
        trainer.logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
