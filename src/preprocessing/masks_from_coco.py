"""
Convert COCO JSON annotations (from digitalsreeni_image_annotator) to
georeferenced GeoTIFF semantic masks.

The annotation GUI works in pure pixel space; geographic reference is
preserved in the original Sentinel-2 GeoTIFF.  This script:

  1. Reads the COCO JSON exported from the GUI (pixel-coordinate polygons).
  2. Builds a deterministic class→pixel-value mapping (sorted category names,
     values 1..N; 0 = unannotated background).
  3. Rasterises each class separately into a single uint8 mask, so overlapping
     polygons of different classes are painted in pixel-value order (higher
     priority classes overwrite lower ones).
  4. Writes a single-band GeoTIFF inheriting the CRS and transform of the
     source file → full geographic reference is restored.
  5. Writes a companion 'class_pixel_mapping.txt' next to the mask.

Naming convention assumed:
  Source GeoTIFF  : <date>_S2_<tile>_<site>_ms.tif
  Annotation PNG  : <date>_S2_<tile>_<site>_[ms_]rgb_annotation.png
  Output mask     : <date>_S2_<tile>_<site>_mask_gt.tif
"""

import json
import os
import re

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds, IDENTITY
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Configuration  –  edit these paths to match your project layout
# ---------------------------------------------------------------------------
COCO_JSON_PATH = "data/images/salinas/annotation_masks_proj/annotations.json"
SOURCE_TIFF_DIR = "data/images/salinas/ms"
OUTPUT_MASK_DIR = "data/images/salinas/annotation_masks_gt"
# ---------------------------------------------------------------------------


def _png_name_to_tiff_name(png_filename: str) -> str:
    """
    Derive the source TIFF filename from an annotation PNG filename.

    Handles two naming patterns produced by create_png_for_annotation():
      - '<base>_ms_rgb_annotation.png'  → '<base>_ms.tif'  (full pipeline)
      - '<base>_rgb_annotation.png'     → '<base>_ms.tif'  (RGB-only input)

    Examples:
        '2015-08-01-11-16-33_S2_29TQJ_SALINAS_ms_rgb_annotation.png'
        → '2015-08-01-11-16-33_S2_29TQJ_SALINAS_ms.tif'

        '2015-07-25-11-25-40_S2_29TQJ_SALINAS_rgb_annotation.png'
        → '2015-07-25-11-25-40_S2_29TQJ_SALINAS_ms.tif'
    """
    stem = os.path.splitext(png_filename)[0]                  # drop extension
    # Replace either '_ms_rgb_annotation' or '_rgb_annotation' with '_ms'
    stem = re.sub(r'(_ms)?_rgb_annotation$', '_ms', stem)
    return stem + '.tif'


def coco_to_geotiff_masks(
    coco_json_path: str,
    source_tiff_dir: str,
    output_mask_dir: str,
    target_class: str | None = None,
    class_priority: list[str] | None = None,
    ignore_index: int = 255,
    distance_threshold: float = 2.0,
) -> None:
    """
    Parameters
    ----------
    coco_json_path      : path to the COCO JSON exported from the annotation GUI.
    source_tiff_dir     : directory containing the original multi-band GeoTIFFs.
    output_mask_dir     : directory where output GeoTIFF masks will be written.
    target_class        : if given, produce a binary mask (0/1) for that class only.
                          If None, produce a multi-class semantic mask where each
                          class has a unique pixel value (1..N).
    class_priority      : optional ordered list of class names from lowest to
                          highest painting priority.  Classes later in the list
                          overwrite earlier ones where polygons overlap.
                          Defaults to alphabetical order (lowest pixel value first).
    ignore_index        : pixel value written for unannotated background pixels that
                          lie farther than ``distance_threshold`` from any labeled
                          pixel (intended as a DL training ignore label).  Default 255.
    distance_threshold  : maximum Euclidean distance in pixels for gap-filling.
                          Background pixels (value 0) within this distance of a
                          labeled pixel are snapped to that label (nearest-neighbour,
                          to close small annotation gaps / snapping errors).  Pixels
                          beyond the threshold receive ``ignore_index`` instead.
                          Default 2 (= 20 m at Sentinel-2 10 m resolution).
    """
    os.makedirs(output_mask_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Build look-up tables
    id_to_image = {img['id']: img for img in coco['images']}
    id_to_category = {cat['id']: cat['name'] for cat in coco['categories']}

    # Build a stable class→pixel-value mapping (mirrors export_semantic_labels logic)
    all_class_names = sorted({cat['name'] for cat in coco['categories']})
    if target_class is not None:
        # Binary mask: only the requested class → value 1
        class_to_pixel = {target_class: 1}
    else:
        class_to_pixel = {name: i + 1 for i, name in enumerate(all_class_names)}

    # Painting order: lower pixel-value classes first so higher-value ones win on overlap
    paint_order = class_priority if class_priority else sorted(class_to_pixel, key=lambda c: class_to_pixel[c])

    # Write the class mapping file once (same for all images in this JSON)
    mapping_path = os.path.join(output_mask_dir, 'class_pixel_mapping.txt')
    os.makedirs(output_mask_dir, exist_ok=True)
    with open(mapping_path, 'w') as f:
        f.write('Pixel Value : Class Name\n')
        f.write(f'{ignore_index:<12}: ignore_index (unannotated background > {distance_threshold:.0f} px from any label)\n')
        for name, val in sorted(class_to_pixel.items(), key=lambda x: x[1]):
            f.write(f'{val:<12}: {name}\n')
    print(f'Class mapping written to: {mapping_path}')

    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    processed, skipped = 0, 0

    for image_id, annotations in anns_by_image.items():
        image_info = id_to_image[image_id]
        png_filename = image_info['file_name']
        tiff_filename = _png_name_to_tiff_name(png_filename)
        tiff_path = os.path.join(source_tiff_dir, tiff_filename)

        if not os.path.exists(tiff_path):
            print(f"  [SKIP] Source TIFF not found: {tiff_path}")
            skipped += 1
            continue

        # Read spatial metadata from the source GeoTIFF
        with rasterio.open(tiff_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()

        # Group polygons by class name
        polys_by_class: dict[str, list] = {name: [] for name in class_to_pixel}
        for ann in annotations:
            class_name = id_to_category.get(ann['category_id'])
            if class_name not in class_to_pixel:
                continue
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            for seg in ann['segmentation']:
                coords = list(zip(seg[0::2], seg[1::2]))  # [(col, row), ...]
                if len(coords) >= 3:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)  # attempt auto-repair
                    if poly.is_valid and not poly.is_empty:
                        polys_by_class[class_name].append(poly.__geo_interface__)

        if not any(polys_by_class.values()):
            print(f"  [SKIP] No valid polygons for {png_filename}")
            skipped += 1
            continue

        # Rasterise class by class in paint_order (pixel space, IDENTITY transform).
        # Each class overwrites previous ones where polygons overlap, so classes
        # later in paint_order have higher priority.
        mask = np.zeros((height, width), dtype=np.uint8)
        for class_name in paint_order:
            shapes = [(geom, class_to_pixel[class_name]) for geom in polys_by_class.get(class_name, [])]
            if not shapes:
                continue
            layer = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=IDENTITY,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )
            mask = np.where(layer > 0, layer, mask)

        # -------------------------------------------------------------------
        # Gap-filling: resolve all background pixels (value == 0)
        # -------------------------------------------------------------------
        # For every 0-pixel, distance_transform_edt finds the nearest labeled
        # pixel (1..N) and its coordinates.
        #
        #   distance <= distance_threshold  →  snap to that neighbor's label
        #                                       (fixes small gaps / snapping errors)
        #   distance >  distance_threshold  →  assign ignore_index (255)
        #                                       (large unannotated border regions)
        # -------------------------------------------------------------------
        bg_mask = mask == 0
        if np.any(bg_mask):
            distances, indices = distance_transform_edt(
                bg_mask, return_distances=True, return_indices=True
            )
            row_idx, col_idx = indices[0], indices[1]
            nearest_label = mask[row_idx, col_idx]  # label of nearest annotated px

            close_bg = bg_mask & (distances <= distance_threshold)
            far_bg   = bg_mask & (distances >  distance_threshold)
            mask[close_bg] = nearest_label[close_bg]
            mask[far_bg]   = ignore_index

        # Write output GeoTIFF with the original spatial reference
        out_stem = os.path.splitext(tiff_filename)[0].replace('_ms', '')
        out_filename = out_stem + '_mask_gt.tif'
        out_path = os.path.join(output_mask_dir, out_filename)

        profile.update(
            count=1,
            dtype=rasterio.uint8,
            compress='lzw',
            nodata=None,
        )
        # Remove tiling/block params that may clash with single-band output
        for key in ('tiled', 'blockxsize', 'blockysize'):
            profile.pop(key, None)

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mask, 1)

        labeled_px  = int(np.sum((mask > 0) & (mask != ignore_index)))
        ignore_px   = int(np.sum(mask == ignore_index))
        class_counts = {name: int(np.sum(mask == val)) for name, val in class_to_pixel.items()}
        counts_str = ', '.join(f'{n}={v}px' for n, v in class_counts.items() if v > 0)
        print(f"  [OK] {out_filename}  (labeled: {labeled_px}px | ignore: {ignore_px}px | {counts_str})")
        processed += 1

    print(f"\nDone. {processed} masks written, {skipped} images skipped.")
    print(f"Output directory: {os.path.abspath(output_mask_dir)}")


if __name__ == "__main__":
    coco_to_geotiff_masks(
        coco_json_path=COCO_JSON_PATH,
        source_tiff_dir=SOURCE_TIFF_DIR,
        output_mask_dir=OUTPUT_MASK_DIR,
        target_class=None,        # None = multi-class semantic mask
        # class_priority defines paint order: last class wins on overlap.
        # Default is alphabetical (lowest pixel value first).
        # Uncomment and reorder to change priority, e.g. foam over water:
        class_priority=['water', 'sand', 'wet_sand', 'white_water', 'clouds'],
    )
