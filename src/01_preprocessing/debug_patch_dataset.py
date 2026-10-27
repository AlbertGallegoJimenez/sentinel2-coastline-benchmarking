"""
Debug helper: inspect patch grid shapefiles and verify they align with the raster data.
Run from the project root directory:
    python src/01_preprocessing/debug_patch_dataset.py
"""
from pathlib import Path
import glob

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# --- Inspect patch grid shapefile ---
shp_path = DATA_ROOT / "coastal_rois_for_patches" / "puntal" / "roi_puntal_128_128_patches.shp"
shp = gpd.read_file(shp_path)
print("Geometries in puntal patch-grid shapefile:")
print(shp.head())

# --- Verify alignment with a sample MS raster ---
ms_files = sorted((DATA_ROOT / "images" / "puntal" / "ms").glob("*.tif"))
if ms_files:
    ms_path = ms_files[0]
    with rasterio.open(ms_path) as src:
        geom = [shp.iloc[0].geometry]
        out_image, out_transform = rio_mask(src, geom, crop=True)
        print("Cropped MS image shape:", out_image.shape)
else:
    print("No MS .tif files found in data/images/puntal/ms/")
