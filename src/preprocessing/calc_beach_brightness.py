"""
Compute ROI-constrained coastal brightness statistics for Salinas and Puntal.

This script compares two image sources:
1) annotation_pngs: 8-bit PNGs used for annotation.
2) rgb: georeferenced RGB GeoTIFFs.

Filtering rules:
- Use only pixels inside ROI from `data/rois_test_salinas_puntal`.
- Use only images whose acquisition datetime rounded to hour satisfies:
  hs < 0.8, tp < 9, tide < 2.5.

Outputs are written to `results/` for each source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from PIL import Image
from rasterio.features import rasterize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITES = ("salinas", "puntal")
SOURCES = ("annotation_pngs", "rgb")

ROIS_FOLDER = Path("data/rois_test_salinas_puntal")
IMAGES_BASE_FOLDER = Path("data/images")
MARINE_BASE_FOLDER = Path("data/marine_data")
OUT_FOLDER = Path("results")

DATE_FMT = "%Y-%m-%d-%H-%M-%S"

# Marine thresholds requested by user
HS_MAX = 0.8
TP_MAX = 9.0
TIDE_MAX = 2.5


@dataclass
class SitePaths:
	site: str
	source: str
	roi_shp: Path
	images_dir: Path
	marine_csv: Path
	rgb_tif_dir: Path


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _build_site_paths(site: str, source: str) -> SitePaths:
	return SitePaths(
		site=site,
		source=source,
		roi_shp=ROIS_FOLDER / f"roi_{site}.shp",
		images_dir=IMAGES_BASE_FOLDER / site / source,
		marine_csv=MARINE_BASE_FOLDER / site / "marine_data.csv",
		rgb_tif_dir=IMAGES_BASE_FOLDER / site / "rgb",
	)



def _parse_image_dt_hour(filename: str) -> pd.Timestamp:
	"""Parse image datetime and set minute/second/microsecond to zero."""
	dt = datetime.strptime(filename.split("_S2")[0], DATE_FMT)
	dt = dt.replace(minute=0, second=0, microsecond=0)
	return pd.Timestamp(dt)



def _load_filtered_marine_data(marine_csv: Path) -> pd.DataFrame:
	marine_data = pd.read_csv(marine_csv)
	marine_data["date"] = pd.to_datetime(marine_data["date"])

	for col in ("hs", "tp", "tide"):
		marine_data[col] = pd.to_numeric(marine_data[col], errors="coerce")

	valid = (
		marine_data["hs"].lt(HS_MAX)
		& marine_data["tp"].lt(TP_MAX)
		& marine_data["tide"].lt(TIDE_MAX)
	)
	marine_filtered = marine_data.loc[valid, ["date", "hs", "tp", "tide"]].copy()
	marine_filtered.set_index("date", inplace=True)
	return marine_filtered



def _load_roi_gdf(roi_shp: Path) -> gpd.GeoDataFrame:
	gdf = gpd.read_file(roi_shp)
	if gdf.empty:
		raise ValueError(f"ROI file has no geometries: {roi_shp}")
	return gdf



def _find_reference_tif_for_png(png_path: Path, rgb_tif_dir: Path) -> Path:
	"""Find the georeferenced RGB TIF matching an annotation PNG."""
	if png_path.name.endswith("_rgb_annotation.png"):
		tif_name = png_path.name.replace("_rgb_annotation.png", "_rgb.tif")
		candidate = rgb_tif_dir / tif_name
		if candidate.exists():
			return candidate

	prefix = png_path.name.split("_S2")[0]
	matches = sorted(rgb_tif_dir.glob(f"{prefix}_S2*_rgb.tif"))
	if not matches:
		raise FileNotFoundError(f"No reference RGB TIF for PNG: {png_path.name}")
	return matches[0]



def _read_rgb_and_georef(image_path: Path, source: str, rgb_tif_dir: Path):
	"""
	Read RGB data and georeference info.

	Returns
	-------
	rgb:
		Array (H, W, 3) as float32.
	transform:
		Raster affine transform.
	crs:
		Raster CRS.
	"""
	if source == "rgb":
		with rio.open(image_path) as src:
			rgb = src.read([1, 2, 3]).astype(np.float32)
			transform = src.transform
			crs = src.crs
		rgb = np.moveaxis(rgb, 0, -1)
		return rgb, transform, crs

	if source == "annotation_pngs":
		with Image.open(image_path) as img:
			rgb = np.asarray(img)
		if rgb.ndim != 3 or rgb.shape[2] < 3:
			raise ValueError(f"Invalid PNG shape for RGB analysis: {image_path.name}")
		rgb = rgb[..., :3].astype(np.float32)

		ref_tif = _find_reference_tif_for_png(image_path, rgb_tif_dir)
		with rio.open(ref_tif) as src:
			transform = src.transform
			crs = src.crs
			height, width = src.height, src.width

		if rgb.shape[0] != height or rgb.shape[1] != width:
			raise ValueError(
				f"PNG/TIF size mismatch for {image_path.name}: "
				f"PNG={rgb.shape[1]}x{rgb.shape[0]}, TIF={width}x{height}"
			)

		return rgb, transform, crs

	raise ValueError(f"Unsupported source: {source}")



def _build_roi_mask(roi_gdf: gpd.GeoDataFrame, raster_crs, transform, height: int, width: int) -> np.ndarray:
	"""Rasterize ROI geometry onto the raster pixel grid."""
	roi_proj = roi_gdf.to_crs(raster_crs) if roi_gdf.crs != raster_crs else roi_gdf
	if hasattr(roi_proj.geometry, "union_all"):
		roi_geom = roi_proj.geometry.union_all()
	else:
		roi_geom = roi_proj.geometry.unary_union

	mask = rasterize(
		[(roi_geom, 1)],
		out_shape=(height, width),
		transform=transform,
		fill=0,
		all_touched=False,
		dtype=np.uint8,
	)
	return mask.astype(bool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _extract_roi_metrics(rgb: np.ndarray, roi_mask: np.ndarray) -> dict | None:
	if not roi_mask.any():
		return None

	r = rgb[..., 0]
	g = rgb[..., 1]
	b = rgb[..., 2]

	mean_rgb_px = (r + g + b) / 3.0
	luminance_px = 0.2126 * r + 0.7152 * g + 0.0722 * b
	r_minus_b_px = r - b

	mean_rgb_roi = mean_rgb_px[roi_mask]
	luminance_roi = luminance_px[roi_mask]
	r_minus_b_roi = r_minus_b_px[roi_mask]

	if mean_rgb_roi.size == 0:
		return None

	return {
		"roi_pixels": int(mean_rgb_roi.size),
		"mean_rgb_sum": float(np.sum(mean_rgb_roi)),
		"mean_rgb_sum_sq": float(np.sum(np.square(mean_rgb_roi))),
		"luminance_sum": float(np.sum(luminance_roi)),
		"luminance_sum_sq": float(np.sum(np.square(luminance_roi))),
		"r_minus_b_mean": float(np.mean(r_minus_b_roi)),
		"mean_rgb_image_mean": float(np.mean(mean_rgb_roi)),
		"mean_rgb_image_std": float(np.std(mean_rgb_roi)),
		"luminance_image_mean": float(np.mean(luminance_roi)),
		"luminance_image_std": float(np.std(luminance_roi)),
	}



def _aggregate_metric(sum_vals: float, sum_sq_vals: float, n: int) -> tuple[float, float]:
	if n == 0:
		return np.nan, np.nan
	mean_val = sum_vals / n
	var_val = max((sum_sq_vals / n) - (mean_val * mean_val), 0.0)
	return float(mean_val), float(np.sqrt(var_val))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def run_site(site_paths: SitePaths) -> pd.DataFrame:
	if not site_paths.roi_shp.exists():
		raise FileNotFoundError(f"ROI file not found: {site_paths.roi_shp}")
	if not site_paths.images_dir.exists():
		raise FileNotFoundError(f"Images folder not found: {site_paths.images_dir}")
	if not site_paths.marine_csv.exists():
		raise FileNotFoundError(f"Marine CSV not found: {site_paths.marine_csv}")

	marine_ok = _load_filtered_marine_data(site_paths.marine_csv)
	roi_gdf = _load_roi_gdf(site_paths.roi_shp)

	if site_paths.source == "rgb":
		img_files = sorted(site_paths.images_dir.glob("*.tif"))
	else:
		img_files = sorted(site_paths.images_dir.glob("*.png"))

	rows = []
	for img_path in img_files:
		img_dt_hour = _parse_image_dt_hour(img_path.name)
		if img_dt_hour not in marine_ok.index:
			continue

		try:
			rgb, transform, raster_crs = _read_rgb_and_georef(
				img_path,
				source=site_paths.source,
				rgb_tif_dir=site_paths.rgb_tif_dir,
			)
		except (FileNotFoundError, ValueError) as exc:
			print(f"Skipping {img_path.name}: {exc}")
			continue

		height, width = rgb.shape[:2]
		roi_mask = _build_roi_mask(roi_gdf, raster_crs, transform, height, width)
		metrics = _extract_roi_metrics(rgb, roi_mask)
		if metrics is None:
			continue

		rows.append(
			{
				"source": site_paths.source,
				"site": site_paths.site,
				"image_name": img_path.name,
				"image_datetime": img_dt_hour,
				"hs": float(marine_ok.loc[img_dt_hour, "hs"]),
				"tp": float(marine_ok.loc[img_dt_hour, "tp"]),
				"tide": float(marine_ok.loc[img_dt_hour, "tide"]),
				**metrics,
			}
		)

	return pd.DataFrame(rows)



def summarize(per_image_df: pd.DataFrame) -> pd.DataFrame:
	if per_image_df.empty:
		return pd.DataFrame(
			[
				{
					"source": "unknown",
					"site": "all",
					"n_images": 0,
					"n_pixels": 0,
					"mean_rgb_mean": np.nan,
					"mean_rgb_std": np.nan,
					"luminance_mean": np.nan,
					"luminance_std": np.nan,
					"r_minus_b_mean": np.nan,
				}
			]
		)

	def _aggregate(group: pd.DataFrame, source: str, site: str) -> dict:
		n_pixels = int(group["roi_pixels"].sum())
		mean_rgb_mean, mean_rgb_std = _aggregate_metric(
			float(group["mean_rgb_sum"].sum()),
			float(group["mean_rgb_sum_sq"].sum()),
			n_pixels,
		)
		luminance_mean, luminance_std = _aggregate_metric(
			float(group["luminance_sum"].sum()),
			float(group["luminance_sum_sq"].sum()),
			n_pixels,
		)
		return {
			"source": source,
			"site": site,
			"n_images": int(group.shape[0]),
			"n_pixels": n_pixels,
			"mean_rgb_mean": mean_rgb_mean,
			"mean_rgb_std": mean_rgb_std,
			"luminance_mean": luminance_mean,
			"luminance_std": luminance_std,
			"r_minus_b_mean": float(group["r_minus_b_mean"].mean()),
		}

	rows = []
	for source, source_group in per_image_df.groupby("source"):
		for site, group in source_group.groupby("site"):
			rows.append(_aggregate(group, source, site))
		rows.append(_aggregate(source_group, source, "all"))

	return pd.DataFrame(rows)



def main() -> None:
	all_frames = []
	for source in SOURCES:
		for site in SITES:
			site_df = run_site(_build_site_paths(site, source))
			all_frames.append(site_df)
			print(
				f"source={source} | site={site}: selected "
				f"{len(site_df)} image(s) after marine filtering."
			)

	per_image_df = pd.concat(all_frames, ignore_index=True)
	summary_df = summarize(per_image_df)

	OUT_FOLDER.mkdir(parents=True, exist_ok=True)

	for source in SOURCES:
		source_per_image = per_image_df[per_image_df["source"] == source].copy()
		source_summary = summary_df[summary_df["source"] == source].copy()

		per_image_out = OUT_FOLDER / f"beach_brightness_per_image_{source}.csv"
		summary_out = OUT_FOLDER / f"beach_brightness_stats_{source}.csv"

		source_per_image.to_csv(per_image_out, index=False)
		source_summary.to_csv(summary_out, index=False)

		print(f"Saved per-image results: {per_image_out}")
		print(f"Saved summary results: {summary_out}")

	print("\nCombined summary:")
	print(summary_df.to_string(index=False))


if __name__ == "__main__":
	main()
