import argparse
import csv
from pathlib import Path
import random
from collections import defaultdict


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

    return parser.parse_args()


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


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.split_name}_train.txt"
    val_path = args.output_dir / f"{args.split_name}_val.txt"
    class_names_path = args.output_dir / "class_names.csv"

    random.seed(args.seed)

    # Collect videos per class
    dataset = defaultdict(list)

    for class_dir in args.dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for video_path in class_dir.glob("*.mp4"):
            dataset[class_name].append(video_path.relative_to(args.dataset_dir))

    print("🔍 Full dataset")
    for name in dataset:
        print(f"   ➡️  {name}: {len(dataset[name])}")
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


    class_mapping = {name: idx for idx, name in enumerate(sorted(dataset.keys()))}

    # Class balanced train/val split
    print("🎯 Train/validation split")
    train_dataset = defaultdict(list)
    val_dataset = defaultdict(list)
    for name in class_mapping.keys():
        videos = dataset[name]
        train_count = min(round(len(videos) * args.train_ratio), len(videos) - 1)
        random.shuffle(videos)
        train_dataset[name] = videos[:train_count]
        val_dataset[name] = videos[train_count:]
        print(
            f"   ➡️  {name}: Train={len(train_dataset[name])}, Val={len(val_dataset[name])}."
        )

    def save_file(dataset, file_path):
        with open(file_path, "w") as f:
            for name in dataset:
                class_id = class_mapping[name]
                lines = [f"{video} {class_id}\n" for video in dataset[name]]
                f.writelines(lines)

    # Scrittura file
    save_file(train_dataset, train_path)
    save_file(val_dataset, val_path)

    with open(class_names_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "name"])
        for name, class_id in class_mapping.items():
            writer.writerow([class_id, name])

    print(f"✅ Created {train_path}, {val_path}, {class_names_path}")


if __name__ == "__main__":
    main()
