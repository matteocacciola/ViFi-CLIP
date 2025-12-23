#!/usr/bin/env python3
"""
Script to export a ViFi-CLIP checkpoint to ONNX format.

Usage:
    python export_to_onnx.py --config <config.yaml> --checkpoint <checkpoint.pth> --output <output_dir>

Example:
    python export_to_onnx.py \
        --config configs/dangers_5_with_freeze.yml \
        --checkpoint .runs/2025-12-17_15-29-30/best.pth \
        --output ./onnx_export
"""

import argparse
import csv
from pathlib import Path
from collections import OrderedDict

import torch
from torch import nn

# Project imports
from trainers.vificlip import returnCLIP
from utils.config import get_config
from utils.logger import create_logger
from utils.tools import export_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export ViFi-CLIP checkpoint to ONNX")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="./onnx_export", help="Output directory")
    return parser.parse_args()


class MockArgs:
    def __init__(self, config_path: str, output_path: str):
        self.config = config_path
        self.output = output_path
        self.opts = None
        self.batch_size = None
        self.pretrained = None
        self.resume = None
        self.accumulation_steps = None
        self.local_rank = 0
        self.only_test = False


def load_class_names(label_list_path: str) -> list[str]:
    with open(label_list_path, "r") as f:
        return [row["name"] for row in csv.DictReader(f)]


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    
    # Remove "module." prefix if present (from DDP training)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Loaded checkpoint: {msg}")
    if "epoch" in checkpoint:
        logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
    
    return model


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(output_dir=output_dir, name="onnx_export")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load configuration
    config = get_config(MockArgs(str(config_path), str(output_dir)))
    
    # Load class names and build model
    class_names = load_class_names(config.DATA.LABEL_LIST)
    logger.info(f"Class names: {class_names}")
    
    model = returnCLIP(config, logger=logger, class_names=class_names)
    model = load_checkpoint_weights(model, str(checkpoint_path), logger)
    model = model.cuda()
    
    # Export to ONNX (now works with unwrapped model)
    export_onnx(model, output_dir, logger)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()