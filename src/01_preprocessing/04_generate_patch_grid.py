"""
Generate a 128×128-pixel patch grid over coastal ROI shapefiles.

For each site shapefile found under ``data/coastal_rois_for_patches/``, a
regular grid of overlapping 128-pixel (1280 m) patches is computed and saved as a
companion ``_128_128_patches.shp`` file alongside the source ROI.

Usage
-----
Run from the project root::

    python src/01_preprocessing/04_generate_patch_grid.py

Configuration
-------------
Adjust the constants below if you need a different patch size or overlap.
All coordinate computations assume a metric CRS (UTM). If the input shapefile
uses a geographic CRS (degrees), the script prints a warning and skips it.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "coastal_rois_for_patches"

PIXEL_RESOLUTION = 10    # Sentinel-2 native resolution (metres)
PATCH_SIZE_PIXELS = 128  # Patch side length in pixels
OVERLAP_PERCENT = 0.15   # 15 % overlap between adjacent patches
MIN_AREA_RATIO = 0.005   # Minimum fraction of patch area that must lie inside the ROI

# Derived spatial parameters
PATCH_SIZE_M = PATCH_SIZE_PIXELS * PIXEL_RESOLUTION
overlap_m = PATCH_SIZE_M * OVERLAP_PERCENT
STRIDE_M = PATCH_SIZE_M - overlap_m


def generate_patch_grid() -> None:
    """Discover ROI shapefiles and write a patch-grid companion for each one."""
    shp_files = sorted(BASE_DIR.glob("**/*.shp"))

    for shp_path in shp_files:
        # Skip already-generated patch files
        if "_patches" in shp_path.name:
            continue

        print(f"-> Processing: {shp_path.relative_to(PROJECT_ROOT)}")
        try:
            gdf = gpd.read_file(shp_path)
        except Exception as exc:
            print(f"   Error loading file: {exc}")
            continue

        if gdf.crs and gdf.crs.is_geographic:
            print(
                "   WARNING: CRS is geographic (degrees). "
                "Re-project to a metric CRS (e.g. UTM) before generating patches. Skipping."
            )
            continue

        all_patches: list = []

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds

            x = minx
            while x + PATCH_SIZE_M <= maxx:
                y = miny
                while y + PATCH_SIZE_M <= maxy:
                    patch_geom = box(x, y, x + PATCH_SIZE_M, y + PATCH_SIZE_M)
                    intersection_area = geom.intersection(patch_geom).area
                    if intersection_area / patch_geom.area >= MIN_AREA_RATIO:
                        all_patches.append(patch_geom)
                    y += STRIDE_M
                x += STRIDE_M

        if all_patches:
            patches_gdf = gpd.GeoDataFrame({"geometry": all_patches}, crs=gdf.crs)
            suffix = f"_{PATCH_SIZE_PIXELS}_{PATCH_SIZE_PIXELS}_patches"
            output_path = shp_path.with_name(shp_path.stem + suffix + ".shp")
            patches_gdf.to_file(output_path)
            print(f"   Created {len(all_patches)} patches → {output_path.relative_to(PROJECT_ROOT)}")
        else:
            print("   No patches met the minimum area-overlap threshold.")


if __name__ == "__main__":
    generate_patch_grid()
