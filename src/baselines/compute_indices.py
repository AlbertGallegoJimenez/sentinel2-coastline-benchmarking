"""
This script computes the following spectral indices for each image in the GT dataset:
- NDWI (Normalized Difference Water Index)
- MNDWI (Modified Normalized Difference Water Index)
- AWEInsh (Automated Water Extraction Index)
The computed indices are saved as .npy files in the `data/indices/{STUDY_NAME}/gt_indices` directory, where `{STUDY_NAME}` is the name of the study area (e.g., "puntal" or "salinas").

The Sentinel-2 bands are stored in the `data/images/{STUDY_NAME}/ms` directory with the following bands:
- Band 1: Blue
- Band 2: Green
- Band 3: Red
- Band 4: NIR
- Band 5: QA

The SWIR1 band are stored as single band .tif files in the `data/images/{STUDY_NAME}/swir` directory. The SWIR images are rescaled to match the spatial resolution
of the Sentinel-2 bands (10m) and are aligned with the RGB images.

The objective of this script is to compute the spectral indices for the GT images, which will be used for benchmarking traditional indices-based methods against DL models trained on the same GT data.
"""

import os
import numpy as np
import rasterio

# Set data paths
STUDY_NAME = "salinas"
MS_DIR = f"data/images/{STUDY_NAME}/ms"
SWIR_DIR = f"data/images/{STUDY_NAME}/swir"
GT_DIR = f"data/images/{STUDY_NAME}/annotation_masks_gt"
OUTPUT_DIR = f"data/indices/{STUDY_NAME}/indices"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_indices(ms_path, swir_path):
    """Compute the spectral indices for a given Sentinel-2 image and its corresponding SWIR image."""
    with rasterio.open(ms_path) as ms_src:
        blue = ms_src.read(1).astype(float)
        green = ms_src.read(2).astype(float)
        red = ms_src.read(3).astype(float)
        nir = ms_src.read(4).astype(float)

    with rasterio.open(swir_path) as swir_src:
        swir1 = swir_src.read(1).astype(float)

    # Compute indices
    ndwi = (green - nir) / (green + nir + 1e-10)
    mndwi = (green - swir1) / (green + swir1 + 1e-10)
    aweinsh = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir1)

    return ndwi, mndwi, aweinsh

# Get list of GT images
gt_images = [file for file in os.listdir(GT_DIR) if file.endswith(".tif")]

# Compute indices for each GT image
for gt_image in gt_images:
    ms_path = os.path.join(MS_DIR, gt_image.replace(".tif", ".png"))
    swir_path = os.path.join(SWIR_DIR, gt_image.replace(".tif", "_swir.tif"))
    
    if not os.path.exists(ms_path) or not os.path.exists(swir_path):
        print(f"Warning: Missing MS or SWIR image for {gt_image}. Skipping.")
        continue

    ndwi, mndwi, aweinsh = compute_indices(ms_path, swir_path)

    # Save indices as .npy files
    np.save(os.path.join(OUTPUT_DIR, gt_image.replace(".tif", "_ndwi.npy")), ndwi)
    np.save(os.path.join(OUTPUT_DIR, gt_image.replace(".tif", "_mndwi.npy")), mndwi)
    np.save(os.path.join(OUTPUT_DIR, gt_image.replace(".tif", "_aweinsh.npy")), aweinsh)