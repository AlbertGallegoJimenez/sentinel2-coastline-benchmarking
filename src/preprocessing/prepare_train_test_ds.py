"""
This script prepares the training and testing datasets for the deep learning models.
It performs the following steps:
1. Loads the GT masks in the `data/images/{STUDY_NAME}/annotation_masks_gt` directory.
2. Loads the corresponding RGB, NIR, SWIR1 and SWIR2 images in the following directories:
   - RGB bands: `data/images/{STUDY_NAME}/rgb`
   - NIR band: `data/images/{STUDY_NAME}/ms` -> In the '..._ms.tif' files, the NIR band is the 4th band (index 3).'
   - SWIR bands: `data/images/{STUDY_NAME}/swir` -> In the '..._swir.tif' files, the SWIR1 band is the 1st band (index 0) and the SWIR2 band is the 2nd band (index 1).'
3. Merges the RGB, NIR, SWIR1 and SWIR2 bands into a single 6-band image.
4. Loads the marine data from the `data/marine_data/{STUDY_NAME}/marine_data.csv` file and extracts the tide level and whether the tide is rising or not for each GT image.
5. Crops the images and masks into 128x128 patches using the shapefiles in `data/coastal_rois_for_patches/{STUDY_NAME}/roi_{STUDY_NAME}_128_128_patches.shp`. Only patches that are exactly 128x128 are kept, to ensure consistency for training.
5. Splits the dataset into training and testing sets, ensuring that the distribution of tide levels and tide states (rising/falling) is similar in both sets. Furthermore, the split is done mixing images from 'salinas' and 'puntal' study areas, to ensure that the models are trained and tested on a diverse set of images.
6. Saves the training and testing datasets in the `data/dl_npy/train` and `data/dl_npy/test` directories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from sklearn.model_selection import train_test_split


DATE_FMT = "%Y-%m-%d-%H-%M-%S"
STUDY_NAMES = ("salinas", "puntal")


@dataclass(frozen=True)
class SampleRecord:
   """Container for a valid sample with all required sources available."""

   sample_id: str
   study_name: str
   timestamp: pd.Timestamp
   gt_path: Path
   rgb_path: Path
   ms_path: Path
   swir_path: Path
   tide: float
   tide_is_rising: bool
   tide_level_quantile: int


def _parse_dt_to_hour_res(filename: str) -> pd.Timestamp:
   """Return acquisition datetime rounded to hourly resolution from file name."""
   dt = pd.to_datetime(filename.split("_S2")[0], format=DATE_FMT)
   return dt.floor("h")


def _calc_tide_is_rising(marine_data: pd.DataFrame) -> pd.Series:
   """Calculate whether tide is rising by positive first-order difference."""
   tide_is_rising = marine_data["tide"].diff() > 0
   return tide_is_rising.fillna(False)


def _index_files_by_timestamp(folder: Path, suffix: str) -> dict[pd.Timestamp, Path]:
   """Index files by hourly timestamp, filtering by expected filename suffix."""
   indexed: dict[pd.Timestamp, Path] = {}
   if not folder.exists():
      return indexed

   for file in folder.iterdir():
      if not file.is_file() or file.suffix.lower() != ".tif":
         continue
      if suffix not in file.name:
         continue
      ts = _parse_dt_to_hour_res(file.name)
      indexed[ts] = file
   return indexed


def _load_marine_data(study_name: str) -> pd.DataFrame:
   """Load marine csv and derive tide-related stratification columns."""
   marine_path = Path(f"data/marine_data/{study_name}/marine_data.csv")
   marine_data = pd.read_csv(marine_path)
   marine_data["date"] = pd.to_datetime(marine_data["date"])
   marine_data = marine_data.set_index("date").sort_index()

   # Keep rows where tide is available.
   marine_data = marine_data[marine_data["tide"].notna()].copy()
   marine_data["tide_is_rising"] = _calc_tide_is_rising(marine_data)
   marine_data["tide_level_quantile"] = pd.qcut(
      marine_data["tide"], q=4, labels=False, duplicates="drop"
   ).astype(int)

   return marine_data


def _build_samples_for_study(study_name: str) -> list[SampleRecord]:
   """Create sample records for one study area, keeping only complete entries."""
   base_dir = Path(f"data/images/{study_name}")
   gt_dir = base_dir / "annotation_masks_gt"
   rgb_dir = base_dir / "rgb"
   ms_dir = base_dir / "ms"
   swir_dir = base_dir / "swir"

   gt_idx = _index_files_by_timestamp(gt_dir, "_mask_gt")
   rgb_idx = _index_files_by_timestamp(rgb_dir, "_rgb")
   ms_idx = _index_files_by_timestamp(ms_dir, "_ms")
   swir_idx = _index_files_by_timestamp(swir_dir, "_swir")
   marine_data = _load_marine_data(study_name)

   records: list[SampleRecord] = []
   for ts, gt_path in sorted(gt_idx.items()):
      rgb_path = rgb_idx.get(ts)
      ms_path = ms_idx.get(ts)
      swir_path = swir_idx.get(ts)
      if rgb_path is None or ms_path is None or swir_path is None:
         continue
      if ts not in marine_data.index:
         continue

      marine_row = marine_data.loc[ts]
      sample_id = f"{study_name}_{ts.strftime('%Y%m%d_%H%M%S')}"
      records.append(
         SampleRecord(
            sample_id=sample_id,
            study_name=study_name,
            timestamp=ts,
            gt_path=gt_path,
            rgb_path=rgb_path,
            ms_path=ms_path,
            swir_path=swir_path,
            tide=float(marine_row["tide"]),
            tide_is_rising=bool(marine_row["tide_is_rising"]),
            tide_level_quantile=int(marine_row["tide_level_quantile"]),
         )
      )

   return records


def _save_split(
   split_name: str,
   records: list[SampleRecord],
   output_root: Path,
) -> None:
   """Crop images/masks using 128x128 patches, save as npy, and write metadata CSV."""
   split_dir = output_root / split_name
   images_dir = split_dir / "images"
   masks_dir = split_dir / "masks"
   images_dir.mkdir(parents=True, exist_ok=True)
   masks_dir.mkdir(parents=True, exist_ok=True)

   metadata_rows = []
   
   # Pre-load shapefiles into a dictionary
   study_shapes = {}
   for study_name in STUDY_NAMES:
       shp_path = Path(f"data/coastal_rois_for_patches/{study_name}/roi_{study_name}_128_128_patches.shp")
       if shp_path.exists():
           study_shapes[study_name] = gpd.read_file(shp_path)
       else:
           print(f"Warning: Shapefile not found at {shp_path}")
           study_shapes[study_name] = None

   for record in records:
      gdf = study_shapes.get(record.study_name)
      if gdf is None:
          continue

      with rasterio.open(record.rgb_path) as src_rgb, \
           rasterio.open(record.ms_path) as src_ms, \
           rasterio.open(record.swir_path) as src_swir, \
           rasterio.open(record.gt_path) as src_gt:

          for idx, row in gdf.iterrows():
              geom = [row.geometry]
              
              try:
                  # Crop by polygon feature
                  rgb, _ = rio_mask(src_rgb, geom, crop=True)
                  ms, _ = rio_mask(src_ms, geom, crop=True)
                  swir, _ = rio_mask(src_swir, geom, crop=True)
                  gt_mask, _ = rio_mask(src_gt, geom, crop=True)
              except ValueError:
                  # Geometry might not intersect with raster entirely
                  continue

              # Extract valid bands
              nir = ms[3:4, ...]      # NIR is 4th band (idx 3)
              swir1 = swir[0:1, ...]  # SWIR1 is 1st band (idx 0)
              swir2 = swir[1:2, ...]  # SWIR2 is 2nd band (idx 1)

              # Filter patches that are smaller than 128x128 (e.g. edge of the map)
              if rgb.shape[1] < 128 or rgb.shape[2] < 128 or gt_mask.shape[1] < 128 or gt_mask.shape[2] < 128:
                  continue

              # Due to floating point bounds in polygon masks, rasterio might return 129x129.
              # We enforce exactly 128x128 by trimming the extra pixels.
              rgb = rgb[:, :128, :128]
              nir = nir[:, :128, :128]
              swir1 = swir1[:, :128, :128]
              swir2 = swir2[:, :128, :128]
              gt_mask = gt_mask[:, :128, :128]

              image = np.concatenate(
                  [rgb, nir, swir1, swir2], axis=0
              ).astype(np.float32)
              mask = gt_mask[0].astype(np.uint8)

              patch_id = f"{record.sample_id}_patch_{idx}"

              image_rel_path = Path("images") / f"{patch_id}.npy"
              mask_rel_path = Path("masks") / f"{patch_id}.npy"

              np.save(split_dir / image_rel_path, image)
              np.save(split_dir / mask_rel_path, mask)

              metadata_rows.append(
                 {
                    "sample_id": patch_id,
                    "study_name": record.study_name,
                    "timestamp": record.timestamp,
                    "tide": record.tide,
                    "tide_is_rising": record.tide_is_rising,
                    "tide_level_quantile": record.tide_level_quantile,
                    "image_npy_path": image_rel_path.as_posix(),
                    "mask_npy_path": mask_rel_path.as_posix(),
                    "source_gt_path": str(record.gt_path),
                    "source_rgb_path": str(record.rgb_path),
                    "source_ms_path": str(record.ms_path),
                    "source_swir_path": str(record.swir_path),
                 }
              )

   if not metadata_rows:
       print(f"Warning: No valid patches generated for {split_name}.")
       return
       
   metadata = pd.DataFrame(metadata_rows).sort_values("sample_id")
   metadata.to_csv(split_dir / "metadata.csv", index=False)

def _build_strata(records: list[SampleRecord]) -> list[str] | None:
   """Build robust strata keys; fallback if some classes are too sparse."""
   options = [
      [
         f"{r.study_name}_{r.tide_level_quantile}_{int(r.tide_is_rising)}"
         for r in records
      ],
      [f"{r.study_name}_{r.tide_level_quantile}" for r in records],
      [r.study_name for r in records],
   ]

   for candidate in options:
      counts = pd.Series(candidate).value_counts()
      if not counts.empty and counts.min() >= 2:
         return candidate
   return None

def main() -> None:
   parser = argparse.ArgumentParser(
      description="Prepare train/test data for DL segmentation from GT Sentinel-2 images."
   )
   parser.add_argument(
      "--test-size",
      type=float,
      default=0.2,
      help="Fraction of samples assigned to test split (default: 0.2).",
   )
   parser.add_argument(
      "--random-state",
      type=int,
      default=42,
      help="Random seed used for train/test split (default: 42).",
   )
   parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("data/dl_npy"),
      help="Output directory where train/test subfolders are written.",
   )
   args = parser.parse_args()

   all_records: list[SampleRecord] = []
   for study_name in STUDY_NAMES:
      records = _build_samples_for_study(study_name)
      print(f"{study_name}: {len(records)} valid samples with all sources + marine data")
      all_records.extend(records)

   if not all_records:
      raise RuntimeError("No valid samples found. Check input folders and marine data coverage.")

   strata = _build_strata(all_records)
   if strata is None:
      print("Warning: strata too sparse, falling back to non-stratified split.")

   train_records, test_records = train_test_split(
      all_records,
      test_size=args.test_size,
      random_state=args.random_state,
      shuffle=True,
      stratify=strata,
   )

   _save_split("train", train_records, args.output_dir)
   _save_split("test", test_records, args.output_dir)

   print("\nSplit summary")
   print(f"Total samples: {len(all_records)}")
   print(f"Train samples: {len(train_records)}")
   print(f"Test samples:  {len(test_records)}")


if __name__ == "__main__":
   main()