from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import torch.multiprocessing as mp

from .dataset import VideoDataset, get_train_transform, get_val_transform

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)

def build_dataloader(config) -> tuple[VideoDataset, VideoDataset, DataLoader, DataLoader]:
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    train_transform = get_train_transform(
        num_frames=config.DATA.NUM_FRAMES,
        input_size=config.DATA.INPUT_SIZE,
        color_jitter=config.AUG.COLOR_JITTER,
        gray_scale=config.AUG.GRAY_SCALE,
    )

    train_data = VideoDataset(
        ann_file=config.DATA.TRAIN_FILE,
        data_root=config.DATA.ROOT,
        labels_file=config.DATA.LABEL_LIST,
        transform=train_transform,
    )
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data,
        sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    val_transform = get_val_transform(
        num_frames=config.DATA.NUM_FRAMES,
        input_size=config.DATA.INPUT_SIZE,
    )
    val_data = VideoDataset(
        ann_file=config.DATA.VAL_FILE,
        data_root=config.DATA.ROOT,
        labels_file=config.DATA.LABEL_LIST,
        transform=val_transform,
    )
    sampler_val = torch.utils.data.DistributedSampler(
        val_data, num_replicas=world_size, rank=global_rank, shuffle=False
    )
    val_loader = DataLoader(
        val_data,
        sampler=sampler_val,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    return train_data, val_data, train_loader, val_loader
