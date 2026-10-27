import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Optional

import numpy as np
import geopandas as gpd
import torch
import rasterio
from rasterio.mask import mask as rasterio_mask
from skimage.measure import find_contours
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import importlib.util as _ilu

# Load models module from the numbered package directory
_models_spec = _ilu.spec_from_file_location(
    "models", Path(__file__).resolve().parent / "models.py"
)
_models_mod = _ilu.module_from_spec(_models_spec)
_models_spec.loader.exec_module(_models_mod)
get_model = _models_mod.get_model

# Configuration
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results/models"
PATCHES_DIR = DATA_ROOT / "coastal_rois_for_patches"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sdw_exports" / "raw"

MIN_LINE_LENGTH_M = 1000.0  # Minimum length filter
STUDY_NAMES = ["puntal", "salinas"]

WATER_CLASSES = [2, 4]

def _index_files(folder: Path, suffix: str) -> dict:
    indexed = {}
    if not folder.exists():
        return indexed
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() == ".tif" and suffix in f.name:
            ts = pd.to_datetime(f.name.split("_S2")[0], format="%Y-%m-%d-%H-%M-%S").floor("h")
            indexed[ts] = f
    return indexed

def get_all_image_records(study_name: str) -> List[Dict]:
    base_dir = DATA_ROOT / "images" / study_name
    rgb_dir = base_dir / "rgb"
    ms_dir = base_dir / "ms"
    swir_dir = base_dir / "swir"
    
    rgb_idx = _index_files(rgb_dir, "_rgb")
    ms_idx = _index_files(ms_dir, "_ms")
    swir_idx = _index_files(swir_dir, "_swir")
    
    records = []
    for ts, rgb_path in sorted(rgb_idx.items()):
        ms_path = ms_idx.get(ts)
        swir_path = swir_idx.get(ts)
        if ms_path and swir_path:
            records.append({
                "timestamp": ts,
                "date_str": ts.strftime("%Y-%m-%d"),
                "rgb_path": rgb_path,
                "ms_path": ms_path,
                "swir_path": swir_path
            })
    return records

def get_model_name_and_params(model_path: Path) -> Tuple[str, str, str]:
    filename = model_path.stem
    parts = filename.replace("best_model_", "").replace("_best_run", "").split("_")
    
    if parts[0] == "deeplabv3plus":
        arch = "deeplabv3plus"
        encoder = "_".join(parts[1:])
    else:
        arch = parts[0]
        encoder = "_".join(parts[1:])
    
    return f"{arch}_{encoder}", arch, encoder

def load_patch_geometries(study_name: str) -> gpd.GeoDataFrame:
    patch_shp = PATCHES_DIR / study_name / f"roi_{study_name}_128_128_patches.shp"
    if not patch_shp.exists():
        raise FileNotFoundError(f"Patch shapefile not found: {patch_shp}")
    return gpd.read_file(patch_shp)

def run_inference_prob(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Sum probabilities of the water classes
    water_prob = probs[WATER_CLASSES[0]] + probs[WATER_CLASSES[1]]
    return water_prob

def extract_contours_from_prob(water_prob: np.ndarray, transform: rasterio.Affine) -> Optional[List[LineString]]:
    # Marching squares directly on probabilities. Level 0.5 interpolates smoothly
    # exactly at the boundary where p(water) = 0.5
    contours = find_contours(water_prob, level=0.5)
    if not contours:
        return None
    
    lines = []
    for contour in contours:
        if contour.shape[0] < 2:
            continue
        coords = [(rasterio.transform.xy(transform, row, col)) for row, col in contour]
        line = LineString(coords)
        if not line.is_empty:
            lines.append(line)
    return lines if lines else None

def load_model(model_path: Path, arch: str, encoder: str, device: torch.device) -> torch.nn.Module:
    model = get_model(arch=arch, encoder_name=encoder)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_files = sorted(glob.glob(str(MODELS_DIR / "best_model_*_best_run.pth")))
    if not model_files:
        print("No best model files found!")
        return
        
    for study_name in STUDY_NAMES:
        print(f"\nProcessing study: {study_name}")
        records = get_all_image_records(study_name)
        print(f"Found {len(records)} full images for {study_name}")
        
        try:
            patches_gdf = load_patch_geometries(study_name)
        except FileNotFoundError as e:
            print(e)
            continue
            
        features = []
        
        for model_file in model_files:
            model_path = Path(model_file)
            model_name, arch, encoder = get_model_name_and_params(model_path)
            print(f"Processing model: {model_name}")
            
            model = load_model(model_path, arch, encoder, device)
            
            for record in records:
                timestamp = record["timestamp"]
                date_str = record["date_str"]
                
                lines_for_image = []
                
                with rasterio.open(record["rgb_path"]) as src_rgb, \
                     rasterio.open(record["ms_path"]) as src_ms, \
                     rasterio.open(record["swir_path"]) as src_swir:
                    
                    for idx, row in patches_gdf.iterrows():
                        geom = [row.geometry]
                        
                        try:
                            rgb, transform = rasterio_mask(src_rgb, geom, crop=True)
                            ms, _ = rasterio_mask(src_ms, geom, crop=True)
                            swir, _ = rasterio_mask(src_swir, geom, crop=True)
                        except ValueError:
                            continue
                            
                        if (rgb.shape[1] < 128 or rgb.shape[2] < 128 or
                            ms.shape[1] < 128 or ms.shape[2] < 128 or
                            swir.shape[1] < 128 or swir.shape[2] < 128):
                            continue
                            
                        rgb = rgb[:, :128, :128]
                        nir = ms[3:4, :128, :128]
                        swir1 = swir[0:1, :128, :128]
                        swir2 = swir[1:2, :128, :128]
                        
                        image = np.concatenate([rgb, nir, swir1, swir2], axis=0).astype(np.float32)
                        
                        water_prob = run_inference_prob(model, image, device)
                        patch_lines = extract_contours_from_prob(water_prob, transform)
                        
                        if patch_lines:
                            lines_for_image.extend(patch_lines)
                            
                if lines_for_image:
                    merged = linemerge(lines_for_image)
                    merged_geoms = [merged] if isinstance(merged, LineString) else list(merged.geoms)
                    
                    final_lines = []
                    for m_geom in merged_geoms:
                        if m_geom.length > MIN_LINE_LENGTH_M:
                            final_lines.append(m_geom)
                            
                    if final_lines:
                        geometry = MultiLineString(final_lines)
                        features.append({
                            "date": date_str,
                            "arch": arch,
                            "encoder": encoder,
                            "model_name": model_name,
                            "geometry": geometry
                        })
        
        if features:
            study_crs = "EPSG:32630" if study_name == "puntal" else "EPSG:32629"
            gdf = gpd.GeoDataFrame(features, crs=study_crs)
            output_path = OUTPUT_DIR / f"sdw_{study_name}.shp"
            gdf.to_file(output_path, driver="ESRI Shapefile")
            print(f"Saved {len(gdf)} features for {study_name} to {output_path}")
        else:
            print(f"No features generated for {study_name}")

if __name__ == "__main__":
    main()
