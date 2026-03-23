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
import matplotlib.pyplot as plt

# Set data paths
STUDY_NAME = "puntal"
MS_DIR = f"data/images/{STUDY_NAME}/ms"
SWIR_DIR = f"data/images/{STUDY_NAME}/swir"
GT_DIR = f"data/images/{STUDY_NAME}/annotation_masks_gt"
OUTPUT_DIR = f"data/images/{STUDY_NAME}/indices"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_tif_filename(gt_image, im_type="ms"):
    """Get the corresponding .tif filename for a given GT image."""
    # To catch the filename in "ms" and "swir" directories, we need to extract the base name from the GT image filename.
    # The GT images are named in the format "YYYY-MM-DD-HH-MM-SS_..._gt.tif", where the date is the first part of the filename.
    # We can use this date to find the corresponding .tif file in the "ms" and "swir" directories.
    date_gt = gt_image.split("_")[0] # The underscore separates the date and the rest of the filename, we take the first part as the date

    if im_type == "ms":
        filename = [f for f in os.listdir(MS_DIR) if f.startswith(date_gt) and f.endswith(".tif")][0]
    else:
        filename = [f for f in os.listdir(SWIR_DIR) if f.startswith(date_gt) and f.endswith(".tif")][0]

    return filename

def compute_indices(ms_path, swir_path):
    """Compute the spectral indices for a given Sentinel-2 image and its corresponding SWIR image."""
    with rasterio.open(ms_path) as ms_src:
        blue = ms_src.read(1).astype(float)
        green = ms_src.read(2).astype(float)
        red = ms_src.read(3).astype(float)
        nir = ms_src.read(4).astype(float)
        profile = ms_src.profile

    with rasterio.open(swir_path) as swir_src:
        swir1 = swir_src.read(1).astype(float)
        swir2 = swir_src.read(2).astype(float) if swir_src.count > 1 else np.zeros_like(swir1)

    # Compute indices
    ndwi = (green - nir) / (green + nir + 1e-10)
    mndwi = (green - swir1) / (green + swir1 + 1e-10)
    aweinsh = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir1)
    scowi = blue + 2 * (green - nir) - 0.75 * swir1 - 0.5 * swir2

    # Set boundaries to NaN for indices using SWIR due to lower resolution artifacts
    mndwi[:2, :] = np.nan
    mndwi[:, :4] = np.nan
    mndwi[:, -3:] = np.nan

    aweinsh[:2, :] = np.nan
    aweinsh[:, :4] = np.nan
    aweinsh[:, -3:] = np.nan

    scowi[:2, :] = np.nan
    scowi[:, :4] = np.nan
    scowi[:, -3:] = np.nan

    profile.update(dtype=rasterio.float32, count=1, driver='GTiff')

    return ndwi, mndwi, aweinsh, scowi, profile

# Get list of GT images
gt_images = [file for file in os.listdir(GT_DIR) if file.endswith(".tif")]

# Compute indices for each GT image
for gt_image in gt_images:
    ms_path = os.path.join(MS_DIR, get_tif_filename(gt_image))
    swir_path = os.path.join(SWIR_DIR, get_tif_filename(gt_image, im_type="swir"))

    if not os.path.exists(ms_path) or not os.path.exists(swir_path):
        print(f"Warning: Missing MS or SWIR image for {gt_image}. Skipping.")
        continue

    ndwi, mndwi, aweinsh, scowi, profile = compute_indices(ms_path, swir_path)

    # Save indices as .tif files
    base_name = gt_image.replace(".tif", "").replace("_mask_gt", "")
    
    with rasterio.open(os.path.join(OUTPUT_DIR, f"{base_name}_ndwi.tif"), 'w', **profile) as dst:
        dst.write(ndwi.astype(rasterio.float32), 1)
    with rasterio.open(os.path.join(OUTPUT_DIR, f"{base_name}_mndwi.tif"), 'w', **profile) as dst:
        dst.write(mndwi.astype(rasterio.float32), 1)
    with rasterio.open(os.path.join(OUTPUT_DIR, f"{base_name}_aweinsh.tif"), 'w', **profile) as dst:
        dst.write(aweinsh.astype(rasterio.float32), 1)
    with rasterio.open(os.path.join(OUTPUT_DIR, f"{base_name}_scowi.tif"), 'w', **profile) as dst:
        dst.write(scowi.astype(rasterio.float32), 1)
        
    print(f"Computed indices for {gt_image} and saved to {OUTPUT_DIR}.")

# # Plot an example of the computed indices
# from rasterio.plot import show

# fig, ax = plt.subplots()

# # Get an example image
# gt_image = [f for f in gt_images if "2016-11-16" in f][0] # Get the GT image for 2016-10-16
# ms_path = os.path.join(MS_DIR, get_tif_filename(gt_image))
# swir_path = os.path.join(SWIR_DIR, get_tif_filename(gt_image, im_type="swir"))
# ndwi, mndwi, aweinsh, profile = compute_indices(ms_path, swir_path)

# # # get the extent from the transform and shape
# transform = profile['transform']
# width = profile['width']
# height = profile['height']
# extent = [transform[2], transform[2] + transform[0] * width,
#           transform[5] + transform[4] * height, transform[5]]

# ax.imshow(mndwi, cmap='viridis_r', extent=extent)
          
# # Remove all axes and ticks
# ax.set_xticks([])
# ax.set_yticks([])
# # Remove borders
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

# fig.savefig(
#     "C:/Users/usuari/Desktop/TFM/figures/spectral_indices_intro/example_mndwi.png",
#     bbox_inches='tight', pad_inches=0, dpi=300)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # get the extent from the transform and shape
# transform = profile['transform']
# width = profile['width']
# height = profile['height']
# extent = [transform[2], transform[2] + transform[0] * width,
#           transform[5] + transform[4] * height, transform[5]]

# im0 = axes[0].imshow(ndwi, cmap='viridis', extent=extent)
# axes[0].set_title("NDWI")
# axes[0].set_xlabel("Eastings [m]")
# axes[0].set_ylabel("Northings [m]")

# im1 = axes[1].imshow(mndwi, cmap='viridis', extent=extent)
# axes[1].set_title("MNDWI")
# axes[1].set_xlabel("Eastings [m]")
# axes[1].set_ylabel("Northings [m]")

# im2 = axes[2].imshow(aweinsh, cmap='viridis', extent=extent)
# axes[2].set_title("AWEInsh")
# axes[2].set_xlabel("Eastings [m]")
# axes[2].set_ylabel("Northings [m]")

# plt.tight_layout()
# plt.show() 