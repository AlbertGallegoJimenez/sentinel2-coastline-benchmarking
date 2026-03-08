""""
This Python script creates RGB composite images from ms bands. It reads the specified bands from
a single multi-band raster (.tif), combines them into an RGB composite, and saves the resulting
image to the output directory. The script uses the rasterio library to handle geospatial data
and numpy for array manipulation.
"""
import os
import rasterio as rio
import numpy as np
from PIL import Image

def create_rgb_composite(input_path, output_dir, band_mapping):
    """
    Creates an RGB composite image from a single multi-band raster file.

    Parameters:
    input_path (str): Path to the input multi-band .tif file.
    output_dir (str): Directory to save the RGB composite image.
    band_mapping (dict): A dictionary mapping RGB channels to their corresponding band indices
                         (1-based) within the multi-band raster.
                         Example: {'R': 4, 'G': 3, 'B': 2}
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the specified bands from the single multi-band raster
    with rio.open(input_path) as src:
        bands = {color: src.read(idx) for color, idx in band_mapping.items()}
        profile = src.profile

    # Stack the bands into an RGB composite
    rgb_composite = np.stack([bands['R'], bands['G'], bands['B']], axis=0)

    # Build output filename from the input filename, replacing '_ms' suffix with '_rgb'
    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_name = basename.replace('_ms', '_rgb') + '.tif'
    output_path = os.path.join(output_dir, out_name)

    # Save the RGB composite image
    profile.update(count=3, dtype=rgb_composite.dtype)
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(rgb_composite[0], 1)  # Write R channel
        dst.write(rgb_composite[1], 2)  # Write G channel
        dst.write(rgb_composite[2], 3)  # Write B channel

    print(f"Saved: {output_path}")


def _stretch_to_uint8(band, lower_pct=2, upper_pct=98):
    """Linearly stretch a band to 0-255 uint8 using percentile clipping."""
    lo, hi = np.percentile(band[band > 0], (lower_pct, upper_pct)) if np.any(band > 0) else (0, 1)
    stretched = np.clip((band.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return (stretched * 255).astype(np.uint8)


def create_png_for_annotation(input_path, output_dir, band_mapping):
    """
    Creates an 8-bit RGB PNG from a multi-band GeoTIFF for use in annotation tools.

    The PNG is a pixel-for-pixel rendering of the GeoTIFF (same dimensions and spatial
    extent), so pixel coordinates in the annotation can be converted back to geographic
    coordinates using the original GeoTIFF's rasterio transform:
        rasterio.transform.xy(transform, rows, cols)

    Bands are percentile-stretched (2nd–98th) to 8-bit.
    PNG is lossless, ensuring exact pixel correspondence with the source TIFF.

    Parameters:
    input_path (str): Path to the input multi-band .tif file.
    output_dir (str): Directory to save the PNG.
    band_mapping (dict): RGB channel → 1-based band index within the raster.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rio.open(input_path) as src:
        r = src.read(band_mapping['R'])
        g = src.read(band_mapping['G'])
        b = src.read(band_mapping['B'])

    rgb_uint8 = np.stack([
        _stretch_to_uint8(r),
        _stretch_to_uint8(g),
        _stretch_to_uint8(b),
    ], axis=-1)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_name = basename.replace('_ms', '_rgb_annotation') + '.png'
    output_path = os.path.join(output_dir, out_name)

    Image.fromarray(rgb_uint8).save(output_path, 'PNG')
    print(f"Saved annotation PNG: {output_path}")
    return output_path


def create_jpg_preview(input_path, output_dir, band_mapping, quality=90):
    """
    Creates a JPG preview image from a single multi-band raster file.

    The bands are percentile-stretched (2nd–98th) to 8-bit for visual inspection.

    Parameters:
    input_path (str): Path to the input multi-band .tif file.
    output_dir (str): Directory to save the JPG preview image.
    band_mapping (dict): RGB channel → 1-based band index within the raster.
    quality (int): JPEG quality (1–95). Default 90.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rio.open(input_path) as src:
        r = src.read(band_mapping['R'])
        g = src.read(band_mapping['G'])
        b = src.read(band_mapping['B'])

    rgb_uint8 = np.stack([
        _stretch_to_uint8(r),
        _stretch_to_uint8(g),
        _stretch_to_uint8(b),
    ], axis=-1)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_name = basename.replace('_ms', '_rgb') + '.jpg'
    output_path = os.path.join(output_dir, out_name)

    Image.fromarray(rgb_uint8).save(output_path, 'JPEG', quality=quality)
    print(f"Saved preview: {output_path}")


if __name__ == "__main__":

    # Select study site
    site = 'puntal'

    # Define paths
    input_directory = f'data/images/{site}/ms'
    output_directory = f'data/images/{site}/rgb'
    preview_directory = f'data/images/{site}/rgb_preview'       # JPG previews for manual filtering
    annotation_directory = f'data/images/{site}/annotation_pngs' # PNGs for annotation GUI input

    # Band indices (1-based) within the multi-band Sentinel-2 ms raster
    # Sentinel-2: B4=Red, B3=Green, B2=Blue
    band_mapping = {
        'R': 3,  # Red band (B4)
        'G': 2,  # Green band (B3)
        'B': 1,  # Blue band (B2)
    }

    ms_files = [
        f for f in os.listdir(input_directory)
        if f.endswith('.tif') and '_dup' not in f
    ]

    print(f"Found {len(ms_files)} images to process.")
    for file in ms_files:
        input_path = os.path.join(input_directory, file)
        #create_jpg_preview(input_path, preview_directory, band_mapping)
        create_rgb_composite(input_path, output_directory, band_mapping)
        create_png_for_annotation(input_path, annotation_directory, band_mapping)
        