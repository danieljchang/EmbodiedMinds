#!/bin/bash

# Preprocessing script for visual in-context learning with 3D perception

# Step 1: Download raw data if not already present
if [ ! -d "data/raw" ]; then
    echo "Downloading raw data..."
    # Command to download raw data (replace with actual command)
    # wget <URL_TO_RAW_DATA> -P data/raw
fi

# Step 2: Run depth estimation on raw images
echo "Running depth estimation..."
python src/preprocessing/depth_estimation.py --input data/raw --output data/processed/depth_maps

# Step 3: Run object detection on raw images
echo "Running object detection..."
python src/preprocessing/object_detection.py --input data/raw --output data/processed/detections

# Step 4: Apply data augmentation
echo "Applying data augmentation..."
python src/preprocessing/augmentation.py --input data/processed --output data/processed/augmented

echo "Preprocessing completed."