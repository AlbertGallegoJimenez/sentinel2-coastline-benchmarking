"""
Compute spectral water indices (NDWI, MNDWI, AWEInsh, SCOWI) for the annotated
Sentinel-2 images of a given study site and save them as single-band GeoTIFFs.

Output files are written to ``data/images/{STUDY_NAME}/indices/``.

The Sentinel-2 bands expected in the MS rasters are:
  Band 1 – Blue
  Band 2 – Green
  Band 3 – Red
  Band 4 – NIR
  Band 5 – QA

SWIR bands are read from the companion single-band GeoTIFFs in
``data/images/{STUDY_NAME}/swir/`` (SWIR1 = band 1, SWIR2 = band 2 if present).

Usage
-----
Run from the project root for a specific site::

    python src/02_baselines/01_compute_indices.py --site puntal
    python src/02_baselines/01_compute_indices.py --site salinas

Or process both sites sequentially::

    for SITE in puntal salinas; do
        python src/02_baselines/01_compute_indices.py --site $SITE
    done
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute spectral indices for a study site.")
    parser.add_argument(
        "--site",
        required=True,
        choices=["puntal", "salinas"],
        help="Study site name.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tif_filename(gt_image: str, source_dir: Path, im_type: str = "ms") -> str:
    """Return the MS or SWIR filename matching the date prefix of *gt_image*."""
    date_gt = gt_image.split("_")[0]
    matches = [f for f in os.listdir(source_dir) if f.startswith(date_gt) and f.endswith(".tif")]
    if not matches:
        raise FileNotFoundError(f"No {im_type} file found for date prefix '{date_gt}' in {source_dir}")
    return matches[0]


def compute_indices(
    ms_path: Path, swir_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute NDWI, MNDWI, AWEInsh and SCOWI from a Sentinel-2 MS + SWIR pair."""
    with rasterio.open(ms_path) as ms_src:
        blue  = ms_src.read(1).astype(float)
        green = ms_src.read(2).astype(float)
        red   = ms_src.read(3).astype(float)
        nir   = ms_src.read(4).astype(float)
        profile = ms_src.profile

    with rasterio.open(swir_path) as swir_src:
        swir1 = swir_src.read(1).astype(float)
        swir2 = swir_src.read(2).astype(float) if swir_src.count > 1 else np.zeros_like(swir1)

    ndwi   = (green - nir)   / (green + nir   + 1e-10)
    mndwi  = (green - swir1) / (green + swir1 + 1e-10)
    aweinsh = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir1)
    scowi  = blue + 2 * (green - nir) - 0.75 * swir1 - 0.5 * swir2

    # Mask border pixels affected by SWIR resampling artefacts
    for arr in (mndwi, aweinsh, scowi):
        arr[:2, :]  = np.nan
        arr[:, :4]  = np.nan
        arr[:, -3:] = np.nan

    profile.update(dtype=rasterio.float32, count=1, driver="GTiff")
    return ndwi, mndwi, aweinsh, scowi, profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    study_name = args.site

    project_root = Path(__file__).resolve().parent.parent.parent
    ms_dir   = project_root / "data" / "images" / study_name / "ms"
    swir_dir = project_root / "data" / "images" / study_name / "swir"
    gt_dir   = project_root / "data" / "images" / study_name / "annotation_masks_gt"
    out_dir  = project_root / "data" / "images" / study_name / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_images = sorted(f for f in os.listdir(gt_dir) if f.endswith(".tif"))
    print(f"Found {len(gt_images)} GT images for site '{study_name}'.")

    for gt_image in gt_images:
        try:
            ms_path   = ms_dir   / get_tif_filename(gt_image, ms_dir,   "ms")
            swir_path = swir_dir / get_tif_filename(gt_image, swir_dir, "swir")
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}. Skipping {gt_image}.")
            continue

        ndwi, mndwi, aweinsh, scowi, profile = compute_indices(ms_path, swir_path)
        base_name = gt_image.replace(".tif", "").replace("_mask_gt", "")

        for name, arr in [("ndwi", ndwi), ("mndwi", mndwi), ("aweinsh", aweinsh), ("scowi", scowi)]:
            with rasterio.open(out_dir / f"{base_name}_{name}.tif", "w", **profile) as dst:
                dst.write(arr.astype(rasterio.float32), 1)

        print(f"  Computed indices for {gt_image}")

    print(f"Done. Outputs saved to {out_dir.relative_to(project_root)}")


if __name__ == "__main__":
    main()

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