"""
Generate binary water masks from spectral indices and extract Shoreline Detection
Waterlines (SDW) for a given study site.

Binary masks are created by thresholding each index raster and saved as both
``.npy`` and ``.tif`` files under ``data/images/{STUDY_NAME}/binary_masks/``.
Extracted waterlines are written to a GeoPackage in the same directory.

Thresholding methods
--------------------
- ``zero``          – fixed threshold at 0.
- ``otsu``          – standard Otsu threshold computed from valid (finite) pixels.
- ``otsu_modified`` – modified Otsu (Bergsma et al., 2024).

Usage
-----
Run from the project root::

    python src/02_baselines/02_extract_baseline_sdw.py --site puntal
    python src/02_baselines/02_extract_baseline_sdw.py --site salinas
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.signal import find_peaks
from shapely.geometry import LineString, MultiLineString
from skimage.filters import threshold_otsu
from skimage.measure import find_contours


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract baseline SDW waterlines from spectral indices."
    )
    parser.add_argument(
        "--site",
        required=True,
        choices=["puntal", "salinas"],
        help="Study site name.",
    )
    return parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_LAYER = "sdw"
MIN_LINE_LENGTH_M = 500.0


def otsu_modified_threshold(image_data, bins=256):
    """
    Implements the modified Otsu thresholding method from Bergsma et al. (2024).

    This method refines the standard Otsu threshold by finding the local minimum
    of the histogram within an interval defined by the local maxima surrounding
    the initial Otsu value.
    """
    valid_pixels = image_data[np.isfinite(image_data)]

    if valid_pixels.size == 0:
        return None

    t_otsu = threshold_otsu(valid_pixels)

    counts, bin_edges = np.histogram(valid_pixels, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(counts)
    peak_values = bin_centers[peaks]

    left_peaks = peak_values[peak_values <= t_otsu]
    right_peaks = peak_values[peak_values >= t_otsu]

    if left_peaks.size == 0 or right_peaks.size == 0:
        return t_otsu

    t1 = left_peaks[-1]
    t2 = right_peaks[0]

    search_mask = (bin_centers >= t1) & (bin_centers <= t2)
    interval_counts = counts[search_mask]
    interval_bins = bin_centers[search_mask]

    if interval_bins.size == 0:
        return t_otsu

    t_min = interval_bins[np.argmin(interval_counts)]

    return t_min


def create_mask(index_data, threshold):
    """Create binary mask where water pixels are values greater than threshold."""
    valid_mask = np.isfinite(index_data)
    binary_mask = np.zeros(index_data.shape, dtype=np.uint8)
    binary_mask[valid_mask] = (index_data[valid_mask] > threshold).astype(np.uint8)
    return binary_mask


def save_mask(mask, profile, output_path_tif, output_path_npy):
    """Save binary mask as GeoTIFF and NumPy array."""
    mask_profile = profile.copy()
    mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(output_path_tif, "w", **mask_profile) as dst:
        dst.write(mask, 1)

    np.save(output_path_npy, mask)


def get_threshold(index_data, method):
    """Compute threshold using requested method."""
    valid_pixels = index_data[np.isfinite(index_data)]
    if valid_pixels.size == 0:
        return None

    if method == "zero":
        return 0.0
    if method == "otsu":
        return threshold_otsu(valid_pixels)
    if method == "otsu_modified":
        return otsu_modified_threshold(index_data)

    raise ValueError(f"Unknown thresholding method: {method}")


def parse_date_from_base_name(base_name):
    """Extract acquisition datetime (YYYY-MM-DD HH:MM:SS) from index base name."""
    date_time_part = base_name.split("_")[0]
    parts = date_time_part.split("-")

    if len(parts) >= 6:
        return f"{'-'.join(parts[:3])} {':'.join(parts[3:6])}"

    return date_time_part


def extract_sdw_multiline(index_data, threshold, transform):
    """Extract threshold contours, filtering out lines shorter than MIN_LINE_LENGTH_M."""
    filled_data = np.array(index_data, copy=True)
    filled_data[~np.isfinite(filled_data)] = threshold - 1.0

    contours = find_contours(filled_data, level=threshold)
    if not contours:
        return None

    lines = []
    for contour in contours:
        if contour.shape[0] < 2:
            continue
        coords = [xy(transform, row, col) for row, col in contour]
        line = LineString(coords)
        if not line.is_empty and line.length > MIN_LINE_LENGTH_M:
            lines.append(line)

    if not lines:
        return None

    return MultiLineString(lines)


def process_index_file(index_path, sdw_rows):
    """Generate masks for one index file."""
    filename = os.path.basename(index_path)
    base_name = filename.replace(".tif", "")
    index_name = base_name.split("_")[-1].lower()
    date_value = parse_date_from_base_name(base_name)

    with rasterio.open(index_path) as src:
        index_data = src.read(1).astype(float)
        profile = src.profile
        transform = src.transform

    methods = ["zero", "otsu", "otsu_modified"]
    # if index_name == "scowi":
    #     methods.append("otsu_modified")

    for method in methods:
        threshold = get_threshold(index_data, method)
        if threshold is None:
            print(f"Warning: No valid pixels in {filename}. Skipping method '{method}'.")
            continue

        mask = create_mask(index_data, threshold)

        output_name = f"{base_name}_mask_{method}"
        output_tif = os.path.join(OUTPUT_DIR, f"{output_name}.tif")
        output_npy = os.path.join(OUTPUT_DIR, f"{output_name}.npy")
        save_mask(mask, profile, output_tif, output_npy)

        sdw_geometry = extract_sdw_multiline(index_data, threshold, transform)
        if sdw_geometry is not None:
            sdw_rows.append(
                {
                    "date": date_value,
                    "index_type": index_name,
                    "threshold": method,
                    "threshold_value": float(threshold),
                    "mask_file": os.path.basename(output_tif),
                    "geometry": sdw_geometry,
                }
            )

        print(f"Saved {output_name} | threshold={threshold:.6f}")


def main() -> None:
    args = parse_args()
    study_name = args.site

    indices_dir = PROJECT_ROOT / "data" / "images" / study_name / "indices"
    output_dir  = PROJECT_ROOT / "data" / "images" / study_name / "binary_masks"
    output_gpkg = output_dir / "sdw.gpkg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patch process_index_file to use the resolved paths
    global OUTPUT_DIR, OUTPUT_GPKG
    OUTPUT_DIR  = str(output_dir)
    OUTPUT_GPKG = str(output_gpkg)

    if not indices_dir.exists():
        raise FileNotFoundError(f"Indices directory not found: {indices_dir}")

    index_files = sorted(indices_dir.glob("*.tif"))

    if not index_files:
        print(f"No index files found in {indices_dir}.")
        return

    sdw_rows: list = []
    reference_crs = None

    for index_path in index_files:
        with rasterio.open(index_path) as src:
            if reference_crs is None:
                reference_crs = src.crs
        process_index_file(str(index_path), sdw_rows)

    if sdw_rows:
        gdf = gpd.GeoDataFrame(sdw_rows, crs=reference_crs)
        if output_gpkg.exists():
            output_gpkg.unlink()
        gdf.to_file(str(output_gpkg), layer=OUTPUT_LAYER, driver="GPKG")
        print(f"Saved {len(gdf)} SDW features to: {output_gpkg.relative_to(PROJECT_ROOT)}")
    else:
        print("No SDW features were extracted.")

    print(f"All binary masks saved in: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()