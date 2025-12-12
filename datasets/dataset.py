import csv
import math

from functools import cached_property

from pathlib import Path
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torch
from torch import nn
import pandas as pd
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices


def get_train_transform(
    input_size: int,
    color_jitter: float,
    gray_scale: float,
    norm_mean=(123.675, 116.28, 103.53),
    norm_std=(58.395, 57.12, 57.375),
):
    return T.Compose(
        [
            T.RandomResizedCrop(
                size=(input_size, input_size), scale=(0.75, 1.0), ratio=(1, 1)
            ),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=color_jitter,
            ),
            T.RandomGrayscale(p=gray_scale),
            T.ToDtype(torch.float),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


def get_val_transform(
    input_size,
    norm_mean=(123.675, 116.28, 103.53),
    norm_std=(58.395, 57.12, 57.375),
):
    return T.Compose(
        [
            T.Resize(size=input_size),
            T.CenterCrop(size=(input_size, input_size)),
            T.ToDtype(torch.float),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


class VideoDataset(Dataset):
    def __init__(
        self,
        ann_file,
        data_root,
        transform,
        num_frames: int,
        labels_file,
        target_fps: int = 30,
    ):
        super().__init__()
        self.ann_file = ann_file
        self.data_root = Path(data_root)

        self.frame_sampler = SampleFrames(num_frames, target_fps=target_fps)
        self.transform = transform
        self.video_infos = self.load_annotations()

        self.labels_file = labels_file

    @cached_property
    def classes(self):
        with open(self.labels_file) as f:
            reader = csv.DictReader(f)
            classes = [row["name"] for row in reader]
        return classes

    def load_annotations(self):
        """Load annotation file to get video information."""
        df = pd.read_csv(self.ann_file, sep=" ", header=None)

        if len(df.columns) == 2:
            df.columns = ["filename", "label"]
            df["start"] = 0
            df["end"] = np.nan
        elif len(df.columns) == 4:
            df.columns = ["filename", "label", "start", "end"]
        else:
            raise ValueError(f"Expected 2 or 4 columns, got {len(df.columns)}")

        return df

    @cached_property
    def class_probs(self) -> torch.Tensor:
        num_classes = self.video_infos["label"].max() + 1
        weight = torch.zeros(num_classes)
        for label in range(num_classes):
            weight[label] = len(self.video_infos[self.video_infos["label"] == label])

        return weight

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        infos = self.video_infos.iloc[idx]

        sampling_start = int(infos["start"])
        sampling_end = None if np.isnan(infos["end"]) else int(infos["end"])
        video = self.frame_sampler(
            self.data_root / infos["filename"],
            sampling_start=sampling_start,
            sampling_end=sampling_end,
        )
        out = self.transform(video)

        out = {"imgs": out, "label": infos["label"].item()}

        return out


class SampleFrames(nn.Module):
    def __init__(
        self,
        num_frames,
        target_fps=30,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.target_fps = target_fps

    def get_rand_stride(self, video_fps: float):
        ideal_stride = video_fps / self.target_fps
        stride_low = max(1, math.floor(ideal_stride))
        stride_high = max(1, math.ceil(ideal_stride))
        if stride_low == stride_high:
            stride = int(stride_low)
        else:
            fps_low = video_fps / stride_high
            fps_high = video_fps / stride_low
            weight_high = abs(self.target_fps - fps_low)
            weight_low = abs(self.target_fps - fps_high)
            prob_low = weight_low / (weight_low + weight_high)
            stride = stride_low if random.random() < prob_low else stride_high
        return stride

    def forward(
        self,
        filename: str | Path,
        sampling_start: int = 0,
        sampling_end: int | None = None,
    ):
        decoder = VideoDecoder(filename)
        video_fps = decoder.metadata.average_fps
        stride = 1 if video_fps is None else self.get_rand_stride(video_fps)
        clip = clips_at_random_indices(
            decoder,
            num_frames_per_clip=self.num_frames,
            num_indices_between_frames=stride,
            sampling_range_start=sampling_start,
            sampling_range_end=sampling_end,
        )

        return clip.data.squeeze(0)
