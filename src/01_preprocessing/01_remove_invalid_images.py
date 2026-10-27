"""
Remove non-valid Sentinel-2 images from the dataset for a given study site.

The script performs clean-up steps:

1. **Preview deduplication** – scans the RGB-preview folder for JPG thumbnails
   and detects acquisitions that share the same calendar day.  When duplicates
   are found, only the file with the latest acquisition timestamp is kept; all
   older duplicates are deleted from disk.

2. **Multi-band (MS) clean-up** – removes GeoTIFF files in the ms folder that
   are flagged as duplicates (``_dup`` in their filename) *and* whose
   datetime prefix does not appear among the surviving preview files.

3. **Multi-band (MS) clean-up** – removes GeoTIFF files that are not present 
    in the preview folder as they are not valid for training.
    This is a safety check to ensure that only valid MS files are kept, even
    if they were not flagged as duplicates.

Usage
-----
Edit the ``SITE`` constant below (or extend the script with argparse) and run::

    python src/01_preprocessing/01_remove_invalid_images.py
"""

import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE = "puntal"
DATE_FMT = "%Y-%m-%d-%H-%M-%S"

MS_FOLDER = f"data/images/{SITE}/ms"
PREVIEW_FOLDER = f"data/images/{SITE}/rgb_preview"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(filename: str) -> datetime:
    """Return the acquisition :class:`datetime` encoded in *filename*."""
    return datetime.strptime(filename.split("_S2")[0], DATE_FMT)


def deduplicate_previews(folder: str) -> set:
    """Remove older duplicate previews that share the same acquisition day.

    Scans *folder* for ``*.jpg`` files.  For every calendar day that has more
    than one acquisition, all files except the one with the latest timestamp
    are deleted.

    Parameters
    ----------
    folder:
        Path to the RGB-preview directory.

    Returns
    -------
    set
        Datetime-prefix strings (``YYYY-MM-DD-HH-MM-SS``) of the surviving
        preview files.
    """
    # Single os.listdir call — build {day: (fname, datetime)} keeping the max
    best: dict = {}  # day -> (fname, datetime)
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
        dt = _parse_dt(fname)
        day = dt.strftime("%Y-%m-%d")
        if day not in best or dt > best[day][1]:
            best[day] = (fname, dt)

    kept: set = {fname for fname, _ in best.values()}

    # Remove every jpg that is not in the winning set
    removed = 0
    for fname in os.listdir(folder):
        if fname.endswith(".jpg") and fname not in kept:
            os.remove(os.path.join(folder, fname))
            print(f"Removed duplicate preview: {fname}")
            removed += 1

    if removed:
        print(
            f"Warning: {removed} duplicate preview(s) removed from '{folder}'. "
            "Only the most recent acquisition per day was kept."
        )

    return {fname.split("_S2")[0] for fname in kept}


def remove_invalid_ms_files(ms_folder: str, valid_prefixes: set) -> None:
    """Delete duplicate-flagged MS GeoTIFFs that have no valid preview.

    A file is considered invalid when it contains ``_dup`` in its name *and*
    its datetime prefix is absent from *valid_prefixes*.

    Parameters
    ----------
    ms_folder:
        Path to the multi-band (ms) GeoTIFF directory.
    valid_prefixes:
        Set of datetime-prefix strings that correspond to valid acquisitions.
    """
    to_remove = [
        f for f in os.listdir(ms_folder)
        if f.endswith(".tif") and "_dup" in f and f.split("_S2")[0] not in valid_prefixes
    ]

    print(f"Found {len(to_remove)} non-valid MS image(s) to remove.")
    for fname in to_remove:
        path = os.path.join(ms_folder, fname)
        os.remove(path)
        print(f"Removed: {path}")

def remove_ms_without_preview(ms_folder: str, valid_prefixes: set) -> None:
    """Delete MS GeoTIFFs that have no corresponding preview.

    A file is considered invalid when its datetime prefix is absent from
    *valid_prefixes*.

    Parameters
    ----------
    ms_folder:
        Path to the multi-band (ms) GeoTIFF directory.
    valid_prefixes:
        Set of datetime-prefix strings that correspond to valid acquisitions.
    """
    to_remove = [
        f for f in os.listdir(ms_folder)
        if f.endswith(".tif") and f.split("_S2")[0] not in valid_prefixes
    ]

    print(f"Found {len(to_remove)} MS image(s) without preview to remove.")
    for fname in to_remove:
        path = os.path.join(ms_folder, fname)
        os.remove(path)
        print(f"Removed: {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    valid_prefixes = deduplicate_previews(PREVIEW_FOLDER)
    remove_invalid_ms_files(MS_FOLDER, valid_prefixes)
    remove_ms_without_preview(MS_FOLDER, valid_prefixes)