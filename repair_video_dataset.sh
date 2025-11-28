#!/bin/bash

# Simple Video Integrity Checker, copies good files and re-encodes corrupt ones.
# Usage: ./repair_video_dataset.sh <source_directory> <destination_directory>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

# Create output directory
mkdir -p "$DEST_DIR"

# Counters
TOTAL=0
CORRUPT=0
FIXED=0

echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo "--------------------------------"

# Use your working file iteration method
mapfile -t video_files < <(find "$SOURCE_DIR" -type f -iname "*.mp4" | sort)
total_files=${#video_files[@]}

for input_file in "${video_files[@]}"; do
    ((TOTAL++))
    
    # Get relative path and create destination path
    output_file=$(echo "$input_file" | sed "s|^$SOURCE_DIR|$DEST_DIR|")
    output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"

    echo "Checking: $input_file"

    # Simple integrity check - redirect errors properly
    errors=$(ffmpeg -v error -i "$input_file" -f null - 2>&1 >/dev/null)

    # Simple integrity check
    if [[ -z "$errors" ]]; then
        # File is good, copy it
        cp "$input_file" "$output_file"
        echo "✅ OK"
    else
        # File is corrupt, re-encode it
        echo "❌ Corrupted - re-encoding..."
        ((CORRUPT++))
        
        echo "Ouptut file: $output_file"
        if ffmpeg -hide_banner -loglevel error -i "$input_file" -c:v libx264 -c:a aac -y "$output_file" 2>/dev/null; then
            echo "✅ Fixed"
            ((FIXED++))
        else
            echo "⚠️ Failed to re-encode"
            rm -f "$output_file"
        fi
    fi
    
    echo "--------------------------------"
done

# Summary
echo ""
echo "========== Summary =========="
echo "Total files: $TOTAL"
echo "Corrupted: $CORRUPT"
echo "Fixed: $FIXED"
echo "Done!"