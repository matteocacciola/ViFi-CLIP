import argparse
import csv
from itertools import chain
import math
from pathlib import Path
import random
from collections import defaultdict
from typing import Final

from torchcodec.decoders import VideoDecoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split video dataset into train/val sets"
    )

    # Required arguments
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        required=True,
        help="Path to dataset directory containing class subdirectories",
    )
    parser.add_argument(
        "--split-name",
        "-n",
        type=str,
        required=True,
        help="Name for the split (e.g., 'split1', 'fold1'). Files will be named <split-name>_train.txt, etc.",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Directory where split files will be saved (default: current directory)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--min-videos",
        type=int,
        default=2,
        help="Minimum number of videos required per class (default: 2)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of videos for training (default: 0.8)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of class names to include (default: None, includes all classes)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per class (default: None, use all)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=150,
        help="Chunk size in frames, to divide the videos. 0 or -1 disables chunking (default: 150)",
    )

    return parser.parse_args()


def get_frame_count(video_path: Path) -> int:
    """Get total frame count using torchcodec."""
    decoder = VideoDecoder(str(video_path))
    return decoder.metadata.num_frames or 0


def get_chunks(total_frames: int, chunk_size: int) -> list[tuple[int, int | None]]:
    """Divide frames into chunks. Last chunk absorbs remainder."""
    if chunk_size <= 0 or total_frames <= chunk_size:
        return [(0, None)]

    chunks = []
    start = 0
    remaining = total_frames
    while start < total_frames and remaining >= chunk_size:
        end = start + chunk_size
        remaining = total_frames - end
        if remaining < chunk_size:
            # Last chunk absorbs remainder
            chunks.append((start, None))
        else:
            chunks.append((start, end))
            start = end

    return chunks


def check_for_spaces(dataset):
    """Check if any video paths contain spaces and issue warnings."""
    videos_with_spaces = []

    for class_name, videos in dataset.items():
        for video_path in videos:
            if " " in str(video_path):
                videos_with_spaces.append(video_path)

    if videos_with_spaces:
        print("\n⚠️  WARNING: Found video paths with spaces!")
        print("   These spaces will interfere with the 'video_path class_id' format.")
        print("   Affected files:")
        for video_path in videos_with_spaces[:10]:  # Show first 10
            print(f"   ➡️  {video_path}")
        if len(videos_with_spaces) > 10:
            print(f"   ... and {len(videos_with_spaces) - 10} more files")

        print("\n   Consider renaming these files to remove spaces before proceeding.")
        print("   Example: 'my video.mp4' → 'my_video.mp4'\n")


def stochastic_round(x : float, rng : random.Random | None = None) -> int:
    floor = math.floor(x)
    frac = x - floor
    rnd = rng.random() if rng is not None else random.random()
    return floor + (rnd < frac)


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.split_name}_train.txt"
    val_path = args.output_dir / f"{args.split_name}_val.txt"
    class_names_path = args.output_dir / "class_names.csv"

    random.seed(args.seed)

    chunking_enabled = args.chunk_size > 0

    # Collect videos per class
    dataset = defaultdict(list)

    extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
    for class_dir in args.dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for video_path in chain.from_iterable(
            class_dir.glob(ext) for ext in extensions
        ):
            dataset[class_name].append(video_path.relative_to(args.dataset_dir))

    print("🔍 Full dataset")
    for class_name in dataset:
        print(f"   ➡️  {class_name}: {len(dataset[class_name])}")
    check_for_spaces(dataset)

    # Filter dataset
    classes_to_include = (
        set(args.classes) if args.classes is not None else set(dataset.keys())
    )
    dataset = {
        name: vids[: args.max_samples]
        for name, vids in dataset.items()
        if len(vids) >= args.min_videos and name in classes_to_include
    }
    if not dataset:
        raise ValueError(
            f"❌ No class with at least {args.min_videos} vidoes has been found."
        )

    video_chunks: dict[Path, list[tuple[int, int | None]]] = {}
    print("📦 Computing chunks...")
    for class_name in dataset:
        for video in dataset[class_name]:
            total_frames = get_frame_count(args.dataset_dir / video)
            video_chunks[video] = get_chunks(total_frames, args.chunk_size)

    class_mapping = {name: idx for idx, name in enumerate(sorted(dataset.keys()))}

    # Class balanced train/val split
    print("🎯 Train/validation split")
    train_dataset = defaultdict(list)
    val_dataset = defaultdict(list)
    
    
    for class_name in class_mapping.keys():
        videos = dataset[class_name]
        random.shuffle(videos)  # Shuffle to break ties
        # Stratify by video length
        videos = sorted(videos, key=lambda v: len(video_chunks[v]))
        
        bin_size = round(max(5, len(videos) / math.sqrt(len(videos))))
        total_seen = 0
        total_train = 0
        for i in range(0, len(videos), bin_size):
            bin_videos = videos[i : i + bin_size]

            bin_n = len(bin_videos)
            total_seen += bin_n
            ideal_total_train = args.train_ratio * total_seen
            delta = ideal_total_train - total_train
            delta = max(0, min(bin_n, delta))

            take_train = stochastic_round(delta)
            random.shuffle(bin_videos)
            train_dataset[class_name] += bin_videos[:take_train]
            val_dataset[class_name] += bin_videos[take_train:]

            total_train += take_train
        
        # Ensure at least one video in val
        if not val_dataset[class_name] and len(train_dataset[class_name]) > 1:
            moved = train_dataset[class_name].pop()
            val_dataset[class_name].append(moved)
        
        if chunking_enabled:
            train_chunks_count = sum(len(video_chunks[v]) for v in train_dataset[class_name])
            val_chunks_count = sum(len(video_chunks[v]) for v in val_dataset[class_name])
            print(
                    f"   ➡️  {class_name}: Train={len(train_dataset[class_name])} ({train_chunks_count} chunks), Val={len(val_dataset[class_name])} ({val_chunks_count} chunks)"
                )
        else:
            print(
                f"   ➡️  {class_name}: Train={len(train_dataset[class_name])}, Val={len(val_dataset[class_name])}"
            )

    def save_file(dataset, file_path):
        with open(file_path, "w") as f:
            for name in dataset:
                class_id = class_mapping[name]
                for video in dataset[name]:
                    if chunking_enabled:
                        for start, end in video_chunks[video]:
                            f.write(f"{video} {class_id} {start} {end}\n")
                    else:
                        f.write(f"{video} {class_id}\n")

    # Scrittura file
    save_file(train_dataset, train_path)
    save_file(val_dataset, val_path)

    with open(class_names_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "name"])
        for class_name, class_id in class_mapping.items():
            writer.writerow([class_id, class_name])

    print(f"✅ Created {train_path}, {val_path}, {class_names_path}")


if __name__ == "__main__":
    main()
