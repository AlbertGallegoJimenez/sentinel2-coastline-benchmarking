# Comparative Evaluation of Satellite Image Segmentation Methods for Coastline Extraction: Spectral Indices vs. Deep Learning

Master's Thesis (TFM) — Data Science, Universitat Oberta de Catalunya (UOC)

This repository benchmarks traditional remote sensing techniques (spectral indices + Otsu thresholding) against modern deep learning architectures (U-Net, DeepLabV3+, SegFormer) for automated shoreline detection using Sentinel-2 Level-2A imagery.

Study sites: **El Puntal** and **Salinas** (both on the Cantabrian coast, macrotidal).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Phase 0 · Raw Data & GIS (Manual)](#phase-0--raw-data--gis-manual)
   - [Phase 1 · Preprocessing & Ground Truth](#phase-1--preprocessing--ground-truth)
   - [Phase 2 · Spectral-Index Baselines](#phase-2--spectral-index-baselines)
   - [Phase 3 · Deep Learning](#phase-3--deep-learning)
   - [Phase 4 · Evaluation & Analysis](#phase-4--evaluation--analysis)
5. [Data Description](#data-description)
6. [Results](#results)

---

## Project Overview

Accurate and automated coastline extraction from satellite imagery is a key challenge in coastal engineering. Classical approaches threshold spectral water indices (NDWI, MNDWI, AWEInsh) with Otsu's method. This project evaluates whether state-of-the-art deep learning segmentation models can overcome their limitations in complex intertidal environments (foam, wet sand, cloud shadows).

**Evaluation strategy:**
- *Pixel-level* — IoU and Dice score computed inside a boundary zone around the shoreline.
- *Geometric* — Cross-shore offset (mean absolute error in metres) measured along shore-normal transects.

---

## Repository Structure

```
.
├── data/
│   ├── images/                        # Satellite imagery by site
│   │   ├── puntal/
│   │   │   ├── ms/                    # Raw Sentinel-2 multispectral GeoTIFFs (6-band)
│   │   │   ├── swir/                  # SWIR1/SWIR2 bands (resampled to 10 m)
│   │   │   ├── rgb/                   # RGB GeoTIFF composites (georeferenced)
│   │   │   ├── rgb_preview/           # JPG thumbnails for visual QC
│   │   │   ├── annotation_pngs/       # 8-bit PNGs used as input to the annotation GUI
│   │   │   ├── annotation_masks_gt/   # Georeferenced GT masks (GeoTIFF, from COCO JSON)
│   │   │   ├── annotation_masks_proj/ # Intermediate projected masks
│   │   │   ├── indices/               # Spectral index rasters (NDWI, MNDWI, AWEInsh, SCOWI)
│   │   │   ├── binary_masks/          # Thresholded binary water masks + SDW GeoPackage
│   │   │   ├── reference_wl/          # Reference waterlines digitised in GIS (ground truth geometry)
│   │   │   ├── meta/                  # Per-image metadata CSV files
│   │   │   ├── mask/                  # Cloud/QA masks
│   │   └── salinas/                   # Same structure as puntal/
│   ├── dl_npy/                        # Deep learning dataset (128×128 patches, NumPy format)
│   │   ├── train/
│   │   │   ├── images/
│   │   │   ├── masks/
│   │   │   └── metadata.csv
│   │   └── test/
│   │       ├── images/
│   │       ├── masks/
│   │       └── metadata.csv
│   ├── coastal_rois_for_patches/      # Shapefiles defining the coastal strip used for patch extraction
│   │   ├── puntal/
│   │   └── salinas/
│   ├── rois_test_salinas_puntal/      # Evaluation ROI polygons (one per site, for metric computation)
│   ├── marine_data/                   # Tide level & wave data (REDMAR / SIMAR)
│   │   ├── puntal/
│   │   └── salinas/
│   └── gis_reference/                 # Reference polygons created in GIS for spatial filtering of SDW lines
│
├── src/
│   ├── utils/                         # Shared utilities (imported by scripts and notebooks)
│   │   └── evaluation_utils.py        # IoU, Dice, boundary zone, offset helpers
│   ├── 01_preprocessing/              # Phase 1 — data cleaning, RGB composites, GT masks, dataset splits
│   │   ├── 01_remove_invalid_images.py
│   │   ├── 02_create_rgb_composites.py
│   │   ├── 03_masks_from_coco.py
│   │   ├── 04_generate_patch_grid.py
│   │   ├── 05_prepare_train_test_ds.py
│   │   ├── check_tide_distribution.py
│   │   ├── verify_patch_dataset.py
│   │   └── debug_patch_dataset.py
│   ├── 02_baselines/                  # Phase 2 — spectral indices and Otsu thresholding
│   │   ├── 01_compute_indices.py
│   │   ├── 02_extract_baseline_sdw.py
│   │   └── plot_spectral_cross_section.py
│   ├── 03_deep_learning/              # Phase 3 — model definitions, training, and SDW extraction
│   │   ├── models.py
│   │   ├── 02_train_experiments.py
│   │   └── 03_extract_dl_sdw.py
│   └── 04_evaluation/                 # Phase 4 — standalone evaluation scripts
│       └── calc_beach_brightness.py
│
├── notebooks/                         # Interactive analysis and visualisation
│   ├── 01_sand_brightness_analysis.ipynb
│   ├── 02_pixel_metrics_evaluation.ipynb
│   ├── 03a_pixel_metrics_dl_binary.ipynb
│   ├── 03b_pixel_metrics_dl_multiclass.ipynb
│   ├── 04_cross_shore_offsets_analysis.ipynb
│   ├── check_exported_sdw.ipynb
│   └── utils/
│       └── evaluation_utils.py        # (symlink / copy — canonical source is src/utils/)
│
├── results/
│   ├── models/                        # Trained model checkpoints (.pth) and training summary CSV
│   ├── metrics/                       # All tabular evaluation results (CSV)
│   ├── offsets/                       # Cross-shore offset CSVs per site and method family
│   ├── sdw_exports/
│   │   ├── raw/                       # SDW shapefiles as produced by the algorithms
│   │   └── filtered/                  # SDW shapefiles after GIS spatial filtering
│   └── figures/
│       ├── sand_reflectance/
│       ├── pixel_evaluation/
│       └── offsets_assessment/
│
├── docs/                              # Thesis documents (ignored in this guide)
├── requirements.txt
└── README.md
```

> **Note on Python imports from numbered directories:**  
> Python module identifiers cannot start with a digit. Scripts inside `src/01_preprocessing/`, `src/02_baselines/`, etc. use `importlib` to load sibling modules at runtime, and add the project root to `sys.path` for `src.utils`. Always run scripts from the **project root**.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

All scripts must be run from the **project root** directory (the folder containing this `README.md`).

---

## Pipeline Walkthrough

### Phase 0 · Raw Data & GIS (Manual)

These steps were performed outside of Python and are not automated:

| Step | Tool | Output location |
|------|------|-----------------|
| Download Sentinel-2 L2A imagery via Google Earth Engine | GEE | `data/images/{site}/ms/`, `data/images/{site}/swir/` |
| Download tide and wave data | REDMAR / SIMAR portal | `data/marine_data/{site}/marine_data.csv` |
| Define coastal strip ROI polygon for patch extraction | QGIS | `data/coastal_rois_for_patches/{site}/roi_{site}.shp` |
| Generate RGB preview thumbnails for visual QC | QGIS / manual inspection | `data/images/{site}/rgb_preview/` |
| Select valid images for annotation (cloud-free, correct tide) | Manual visual inspection | — |
| Digitise reference waterlines (ground truth geometry) | QGIS | `data/images/{site}/reference_wl/` |
| Create spatial filtering polygons for SDW post-processing | QGIS | `data/gis_reference/ref_pol_{site}.*` |

---

### Phase 1 · Preprocessing & Ground Truth

#### Step 1.1 — Remove invalid / duplicate images
```bash
python src/01_preprocessing/01_remove_invalid_images.py
```
Edit the `SITE` constant at the top of the script before running. Cleans duplicate acquisitions and removes MS files not validated by the preview thumbnails.

#### Step 1.2 — Create RGB GeoTIFF composites
```bash
python src/01_preprocessing/02_create_rgb_composites.py
```
Builds georeferenced RGB GeoTIFFs from the multispectral bands. Also generates the 8-bit PNGs (`annotation_pngs/`) used as input to the annotation GUI.

#### Step 1.3 — Annotate images with SAM (Manual, external tool)

Images in `data/images/{site}/annotation_pngs/` were loaded into the **[digitalsreeni Image Annotator](https://github.com/digitalsreeni/image-annotator)** (SAM-assisted polygon labelling). Annotations were exported as a **COCO JSON** file per site.

**Class mapping used:**

| Class | Label | Pixel value |
|-------|-------|-------------|
| Cloud / shadow | `cloud` | 1 |
| Dry sand / land | `dry_sand` | 2 |
| Wet sand / swash | `wet_sand` | 3 |
| Calm water | `water` | 4 |
| Foam / breaking waves | `foam` | 5 |

The COCO JSON output should be placed at `data/images/{site}/annotations.json` before running the next step.

#### Step 1.4 — Convert COCO JSON to georeferenced GT masks
```bash
python src/01_preprocessing/03_masks_from_coco.py
```
Reads the COCO JSON, rasterises polygon annotations into a single-band uint8 GeoTIFF per image (inheriting CRS and geotransform from the source MS file), and writes a `class_pixel_mapping.txt` companion file.

Output: `data/images/{site}/annotation_masks_gt/`

#### Step 1.5 — Generate 128×128 patch grid
```bash
python src/01_preprocessing/04_generate_patch_grid.py
```
Creates a shapefile of overlapping 128-pixel (1280 m) patch bounding boxes covering the coastal ROI. The patch grid is required by the dataset preparation step.

Output: `data/coastal_rois_for_patches/{site}/roi_{site}_128_128_patches.shp`

#### Step 1.6 — Prepare train/test dataset
```bash
python src/01_preprocessing/05_prepare_train_test_ds.py
```
Merges RGB + NIR + SWIR1 + SWIR2 into 6-channel NumPy arrays, crops them to the 128×128 patch grid, and performs a stratified train/test split (balanced by tide level and rising/falling state, mixing both sites).

Output: `data/dl_npy/train/` and `data/dl_npy/test/`

#### (Optional) Verify the dataset
```bash
python src/01_preprocessing/verify_patch_dataset.py
```
Checks that all patches are exactly `(6, 128, 128)` for images and `(128, 128)` for masks.

---

### Phase 2 · Spectral-Index Baselines

#### Step 2.1 — Compute spectral indices
```bash
python src/02_baselines/01_compute_indices.py --site puntal
python src/02_baselines/01_compute_indices.py --site salinas
```
Computes NDWI, MNDWI, AWEInsh and SCOWI from the MS + SWIR rasters for every annotated (GT) image and saves the results as single-band GeoTIFFs.

Output: `data/images/{site}/indices/`

#### Step 2.2 — Threshold indices and extract baseline SDW
```bash
python src/02_baselines/02_extract_baseline_sdw.py --site puntal
python src/02_baselines/02_extract_baseline_sdw.py --site salinas
```
Applies three thresholding strategies (fixed zero, Otsu, modified Otsu) to each index raster. Binary water masks are saved as `.tif` and `.npy`. Waterlines (SDW) are extracted as `LineString` geometries and saved to a GeoPackage.

Output: `data/images/{site}/binary_masks/`, `data/images/{site}/binary_masks/sdw.gpkg`

#### Step 2.3 — Spatial filtering of baseline SDW (Manual, GIS)

The raw SDW lines include spurious segments (e.g. inland water bodies, field boundaries). These were filtered in **QGIS** by spatially intersecting with the reference polygons in `data/gis_reference/ref_pol_{site}.*`.

Filtered shapefiles are exported to `results/sdw_exports/filtered/`.

---

### Phase 3 · Deep Learning

#### Step 3.1 — Train segmentation models
```bash
python src/03_deep_learning/02_train_experiments.py
```
Trains the full grid of architecture × encoder combinations (U-Net, DeepLabV3+, SegFormer with MiT-B2/B4 and EfficientNet-B4/B7 encoders) using the `data/dl_npy/` dataset. Best model checkpoints are saved to `results/models/` and a training summary CSV is written to `results/models/experiment_runs_summary.csv`.

#### Step 3.2 — Extract DL SDW waterlines
```bash
python src/03_deep_learning/03_extract_dl_sdw.py
```
Runs inference with all trained models over the complete (non-patched) test scenes for both sites. Predicted segmentation masks are converted to waterline geometries and saved to `results/sdw_exports/raw/`.

#### Step 3.3 — Spatial filtering of DL SDW (Manual, GIS)

Same filtering step as for the baselines: QGIS intersection with `data/gis_reference/ref_pol_{site}.*`.

Filtered shapefiles → `results/sdw_exports/filtered/`.

---

### Phase 4 · Evaluation & Analysis

All evaluation and figure generation is performed interactively in Jupyter notebooks. Run from the project root:

```bash
jupyter lab notebooks/
```

| Notebook | Description |
|----------|-------------|
| [`01_sand_brightness_analysis.ipynb`](notebooks/01_sand_brightness_analysis.ipynb) | Physical context: sand reflectance and colorimetric assessment across sites |
| [`02_pixel_metrics_evaluation.ipynb`](notebooks/02_pixel_metrics_evaluation.ipynb) | Pixel-level metrics (IoU, Dice) for index-based baselines |
| [`03a_pixel_metrics_dl_binary.ipynb`](notebooks/03a_pixel_metrics_dl_binary.ipynb) | Pixel-level metrics for DL models (binary water/land) |
| [`03b_pixel_metrics_dl_multiclass.ipynb`](notebooks/03b_pixel_metrics_dl_multiclass.ipynb) | Pixel-level metrics for DL models (multiclass) |
| [`04_cross_shore_offsets_analysis.ipynb`](notebooks/04_cross_shore_offsets_analysis.ipynb) | Geometric evaluation: cross-shore offset analysis against reference waterlines |
| [`check_exported_sdw.ipynb`](notebooks/check_exported_sdw.ipynb) | Quick visual inspection of exported SDW lines vs. reference waterlines |

The notebook for beach brightness statistics also relies on:
```bash
python src/04_evaluation/calc_beach_brightness.py
```

---

## Data Description

### Sentinel-2 band layout in MS rasters

| Band index (1-based) | Sentinel-2 band | Wavelength |
|----------------------|-----------------|------------|
| 1 | Blue (B2) | 490 nm |
| 2 | Green (B3) | 560 nm |
| 3 | Red (B4) | 665 nm |
| 4 | NIR (B8) | 842 nm |
| 5 | QA60 (cloud mask) | — |

SWIR bands (B11 ~1610 nm, B12 ~2190 nm) are stored separately in `swir/` resampled to 10 m.

### Marine data columns (marine_data.csv)

| Column | Description |
|--------|-------------|
| `datetime` | UTC acquisition datetime |
| `tide_level` | Instantaneous tide level (metres, chart datum) |
| `hs` | Significant wave height (metres) |
| `tp` | Peak wave period (seconds) |

---

## Results

Trained model weights, evaluation metrics and figures are stored under `results/`:

- **`results/models/`** — `.pth` checkpoints for each architecture/encoder combination
- **`results/metrics/`** — CSV tables with pixel-based and boundary metrics
- **`results/offsets/`** — Cross-shore offset CSVs per site and method family
- **`results/sdw_exports/raw/`** — Raw waterlines from algorithms
- **`results/sdw_exports/filtered/`** — Waterlines after GIS post-processing
- **`results/figures/`** — Publication-quality figures

Full analysis and interpretation can be found in [`docs/final_report/TFM_Gallego_Jimenez_Albert.pdf`](docs/final_report/TFM_Gallego_Jimenez_Albert.pdf).
