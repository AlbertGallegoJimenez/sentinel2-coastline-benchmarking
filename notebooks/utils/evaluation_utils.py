import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import rasterio
from rasterio.mask import mask as rasterio_mask


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_default_images_dir() -> Path:
    """Resolve a sensible default images directory for notebook and script usage."""
    candidates = [
        _REPO_ROOT / "data" / "images",
        _REPO_ROOT / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_IMAGES_DIR = _resolve_default_images_dir()

def collapse_gt_to_binary(
    gt_array: np.ndarray,
    water_classes: List[int] = [3, 5],
    no_water_classes: List[int] = [2, 4],
    ignore_classes: List[int] = [1, 255]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse multi-class GT mask to binary water/no-water with ignore mask.
    
    Parameters
    ----------
    gt_array : np.ndarray
        Multi-class ground truth array
    water_classes : List[int]
        Pixel values representing water
    no_water_classes : List[int]
        Pixel values representing no-water (land)
    ignore_classes : List[int]
        Pixel values to ignore in evaluation
        
    Returns
    -------
    binary_gt : np.ndarray (uint8)
        Binary mask where 1=water, 0=no-water
    ignore_mask : np.ndarray (bool)
        True for pixels that should be ignored
        
    Notes
    -----
    - Pixels with ignore_classes are marked in ignore_mask
    - Pixels NOT in any known class (water, no-water, ignore) are also marked as ignore
    - This prevents creating artificial boundaries with unannotated regions
    - Binary GT only contains valid 0/1 values for explicitly annotated pixels
    """
    binary_gt = np.zeros_like(gt_array, dtype=np.uint8)
    
    # All known/valid classes
    all_known_classes = water_classes + no_water_classes + ignore_classes
    
    # Create ignore mask for:
    # 1. Explicit ignore classes (clouds=1, unannotated background=255)
    # 2. Any pixel NOT in water, no-water, or ignore classes (unannotated pixels like value 0)
    ignore_mask = ~np.isin(gt_array, all_known_classes) | np.isin(gt_array, ignore_classes)
    
    # Mark water pixels as 1
    binary_gt[np.isin(gt_array, water_classes)] = 1
    # No-water pixels remain 0 (only if explicitly in no_water_classes)
    
    return binary_gt, ignore_mask


def create_boundary_zone_mask(
    binary_mask: np.ndarray,
    ignore_mask: np.ndarray,
    offset_pixels: int
) -> np.ndarray:
    """
    Create a symmetric buffer zone around the water/land boundary.
    Only considers explicitly annotated pixels (excludes ignored/unannotated pixels).

    Uses Euclidean Distance Transform (EDT) to identify pixels within
    a specified distance from both water and land classes.

    Parameters
    ----------
    binary_mask : np.ndarray (uint8)
        Binary mask where 1=water, 0=land
    ignore_mask : np.ndarray (bool)
        Mask of pixels to ignore (True = ignore, False = valid)
    offset_pixels : int
        Number of pixels to extend buffer on each side of boundary

    Returns
    -------
    boundary_zone : np.ndarray (bool)
        Boolean mask where True=inside boundary zone, False=outside
        Returns all False if no boundary exists (all water, all land, or insufficient valid pixels)

    Notes
    -----
    - Creates symmetric buffer (equal distance landward and seaward)
    - Uses distance_transform_edt for Euclidean distance (circular buffer)
    - Only valid (non-ignored) pixels are used to compute boundaries
    - This prevents artificial boundaries at edges of unannotated regions
    - Consistent with existing codebase (masks_from_coco.py uses EDT)

    Examples
    --------
    >>> binary_gt = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    >>> ignore_mask = np.zeros_like(binary_gt, dtype=bool)
    >>> boundary = create_boundary_zone_mask(binary_gt, ignore_mask, offset_pixels=1)
    >>> # Returns zone within 1 pixel of water/land boundary
    """
    from scipy.ndimage import distance_transform_edt

    # Create masks for valid water and valid land only
    valid_mask = ~ignore_mask
    water_valid = (binary_mask == 1) & valid_mask
    land_valid = (binary_mask == 0) & valid_mask

    # Handle edge case: no valid boundary exists
    if not (np.any(water_valid) and np.any(land_valid)):
        return np.zeros_like(binary_mask, dtype=bool)

    # Compute distance from valid water pixels
    # EDT of ~water_valid gives distance to nearest valid water pixel
    dist_from_water = distance_transform_edt(~water_valid)

    # Compute distance from valid land pixels
    # EDT of ~land_valid gives distance to nearest valid land pixel
    dist_from_land = distance_transform_edt(~land_valid)

    # Boundary zone = valid pixels within offset distance from BOTH water AND land
    # This ensures we only get boundaries between explicitly annotated water and land
    boundary_zone = (dist_from_water <= offset_pixels) & (dist_from_land <= offset_pixels) & valid_mask

    return boundary_zone


def crop_raster_by_geometry(
    raster_path: Path,
    geometry,
    nodata: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[rasterio.Affine], bool]:
    """
    Crop a raster file to a polygon geometry.
    
    Parameters
    ----------
    raster_path : Path
        Path to raster file
    geometry : shapely.geometry.Polygon
        Polygon to crop to
    nodata : Optional[int]
        NoData value to use
        
    Returns
    -------
    cropped_array : np.ndarray or None
        Cropped raster array (2D), None if empty intersection
    transform : rasterio.Affine or None
        Affine transform of cropped raster, None if empty
    is_empty : bool
        True if intersection is empty or all pixels are nodata
        
    Notes
    -----
    - Returns None arrays if polygon doesn't intersect raster
    - Handles cases where crop results in all-nodata array
    """
    try:
        with rasterio.open(raster_path) as src:
            # Crop to geometry
            cropped, transform = rasterio_mask(
                src, 
                [geometry], 
                crop=True,
                filled=True,
                nodata=nodata if nodata is not None else 0
            )
            
            # Extract first band (should be single-band)
            cropped_array = cropped[0]
            
            # Check if crop is empty or all nodata
            if cropped_array.size == 0:
                return None, None, True
                
            # Check if all pixels are nodata
            if nodata is not None and np.all(cropped_array == nodata):
                return None, None, True
                
            return cropped_array, transform, False
            
    except Exception as e:
        print(f"Warning: Failed to crop {raster_path.name}: {e}")
        return None, None, True


def calculate_metrics(
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    ignore_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate IoU and Dice coefficient, excluding ignored pixels.
    
    Parameters
    ----------
    gt_binary : np.ndarray
        Ground truth binary mask (0/1)
    pred_binary : np.ndarray
        Predicted binary mask (0/1)
    ignore_mask : np.ndarray (bool)
        Mask of pixels to ignore (True = ignore)
        
    Returns
    -------
    metrics : Dict[str, float]
        Dictionary with 'iou', 'dice', and 'valid_pixel_count'
        Returns NaN values if no valid pixels exist
        
    Notes
    -----
    - IoU = Intersection / Union
    - Dice = 2 * Intersection / (|A| + |B|)
    - Both metrics exclude pixels where ignore_mask is True
    """
    # Create valid pixel mask (not ignored)
    valid_mask = ~ignore_mask
    
    # Count valid pixels
    valid_pixel_count = np.sum(valid_mask)
    
    # If no valid pixels, return NaN
    if valid_pixel_count == 0:
        return {
            'iou': np.nan,
            'dice': np.nan,
            'valid_pixel_count': 0
        }
    
    # Apply valid mask to both arrays
    gt_valid = gt_binary[valid_mask]
    pred_valid = pred_binary[valid_mask]
    
    # Calculate intersection and union
    intersection = np.sum((gt_valid == 1) & (pred_valid == 1))
    union = np.sum((gt_valid == 1) | (pred_valid == 1))
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    # Calculate Dice
    gt_sum = np.sum(gt_valid == 1)
    pred_sum = np.sum(pred_valid == 1)
    dice = (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'valid_pixel_count': int(valid_pixel_count)
    }


def get_binary_mask_path(
    gt_filename: str,
    study_name: str,
    index: str,
    threshold: str,
    images_dir: Optional[Path] = None,
) -> Path:
    """
    Construct binary mask path from GT filename, index, and threshold.
    
    Parameters
    ----------
    gt_filename : str
        GT mask filename, e.g., '2016-06-26-11-12-58_S2_30TVP_PUNTAL_mask_gt.tif'
    study_name : str
        Study area name ('puntal' or 'salinas')
    index : str
        Spectral index name ('ndwi', 'mndwi', 'aweinsh', 'scowi')
    threshold : str
        Threshold method ('zero', 'otsu', 'otsu_modified')
    images_dir : Optional[Path]
        Base directory containing study folders and binary masks.
        If None, uses DEFAULT_IMAGES_DIR resolved from repository layout.
        
    Returns
    -------
    mask_path : Path
        Path to binary mask file
        
    Notes
    -----
    GT filename format:    YYYY-MM-DD-HH-MM-SS_S2_{tile}_{SITE}_mask_gt.tif
    Binary mask format:    YYYY-MM-DD-HH-MM-SS_S2_{tile}_{SITE}_{index}_mask_{threshold}.tif
    """
    # Remove '_mask_gt.tif' suffix
    base_name = gt_filename.replace('_mask_gt.tif', '')
    
    # Construct binary mask filename
    mask_filename = f"{base_name}_{index}_mask_{threshold}.tif"
    
    # Construct full path
    base_images_dir = Path(images_dir) if images_dir is not None else DEFAULT_IMAGES_DIR
    mask_path = base_images_dir / study_name / "binary_masks" / mask_filename
    
    return mask_path


def parse_gt_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from GT filename.
    
    Parameters
    ----------
    filename : str
        GT filename, e.g., '2016-06-26-11-12-58_S2_30TVP_PUNTAL_mask_gt.tif'
        
    Returns
    -------
    metadata : Dict[str, str]
        Dictionary with 'date', 'time', 'tile', 'site'
        
    Notes
    -----
    Format: YYYY-MM-DD-HH-MM-SS_S2_{tile}_{SITE}_mask_gt.tif
    """
    # Remove extension and suffix
    name = filename.replace('_mask_gt.tif', '')
    
    # Split by underscore
    parts = name.split('_')
    
    # Extract date and time
    datetime_parts = parts[0].split('-')
    date = f"{datetime_parts[0]}-{datetime_parts[1]}-{datetime_parts[2]}"
    time = f"{datetime_parts[3]}:{datetime_parts[4]}:{datetime_parts[5]}"
    
    # Extract tile and site
    tile = parts[2] if len(parts) > 2 else "unknown"
    site = parts[3].upper() if len(parts) > 3 else "unknown"
    
    return {
        'date': date,
        'time': time,
        'tile': tile,
        'site': site
    }