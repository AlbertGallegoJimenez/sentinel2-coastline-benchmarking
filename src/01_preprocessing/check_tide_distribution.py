"""
This script extracts the tide level and whether the tide is rising or not for the GT images of the desired study area.
The GT images are stored in the `data/images/{STUDY_NAME}/annotation_masks_gt` directory, and the marine data is stored
in the `data/marine_data/{STUDY_NAME}/marine_data.csv` file.
The script calculates two columns for the marine data:
(1) a boolean column indicating whether the tide is rising or not,
and (2) a column with the tide level expressed as quantiles of the tide level distribution.
Finally, it shows the distribution of these two columns in the GT data.

The objective of this script is to determine if I'm biasing the selected images for the GT data
towards a specific tide level or towards rising/falling tides, which could affect the performance of the models trained on this data.
The intention is to ensure that the GT data is representative of the overall distribution of tide levels and tide states (rising/falling) in the study area,
not only for the training of DL models but for a fair benchmarking with traditional indices-based methods.
"""

import os
import pandas as pd
from datetime import datetime

# Set data paths
STUDY_NAME = "puntal"
MARINE_PATH = f"data/marine_data/{STUDY_NAME}/marine_data.csv"
GT_DIR = f"data/images/{STUDY_NAME}/annotation_masks_gt"
RGB_TIFF_DIR = f"data/images/{STUDY_NAME}/annotation_pngs"
DATE_FMT = "%Y-%m-%d-%H-%M-%S"

# Helpers
def _parse_dt_to_hour_res(filename: str) -> datetime:
    """Return the acquisition :class:`datetime` encoded in *filename*."""
    strp_time = datetime.strptime(filename.split("_S2")[0], DATE_FMT)

    return strp_time.strftime("%Y-%m-%d %H:00:00")

def _calc_tide_is_rising(marine_data):
    """
    Calculate if the tide is rising or not
    """
    tide_is_rising = marine_data["tide"].diff() > 0
    tide_is_rising = tide_is_rising.fillna(False)
    return tide_is_rising

# Load marine data from csv
marine_data = pd.read_csv(MARINE_PATH)
marine_data["date"] = pd.to_datetime(marine_data["date"])
marine_data.set_index("date", inplace=True)

# Calculate two tide columns: 
# (1) boolean column indicating whether the tide is flood or ebb, 
# and (2) a column with the tide level expressed as quantiles of the tide level distribution
marine_data["tide_is_rising"] = _calc_tide_is_rising(marine_data)
marine_data["tide_level_quantile"] = pd.qcut(marine_data["tide"], q=4, labels=False)

# Get the list of datetime arrays of the GT images
gt_dt = [_parse_dt_to_hour_res(file) for file in os.listdir(GT_DIR) if file.endswith(".tif")]

# Filter the marine_data with the gt_dt array, skipping dates not covered by marine data
covered_gt_dt = [dt for dt in gt_dt if dt in marine_data.index]
skipped_gt = len(gt_dt) - len(covered_gt_dt)
if skipped_gt:
    print(f"Warning: {skipped_gt} GT image(s) have dates not covered by marine data and will be skipped.")
marine_data_gt = marine_data.loc[covered_gt_dt]

# Show the distribution of the tide level quantiles in the GT data
print(f"Distribution of tide level quantiles in the GT data - {STUDY_NAME}:")
print(marine_data_gt["tide_level_quantile"].value_counts())
print(f"Distribution of tide is rising in the GT data - {STUDY_NAME}:")
print(marine_data_gt["tide_is_rising"].value_counts())

# Select N random samples of 3th and 4th quantiles of the tide level distribution
# that are not included in GT
all_dt = [_parse_dt_to_hour_res(file) for file in os.listdir(RGB_TIFF_DIR) if file.endswith(".png")]
not_gt_dt = list(set(all_dt)-set(gt_dt))
covered_not_gt_dt = [dt for dt in not_gt_dt if dt in marine_data.index]
skipped_not_gt = len(not_gt_dt) - len(covered_not_gt_dt)
if skipped_not_gt:
    print(f"Warning: {skipped_not_gt} non-GT image(s) have dates not covered by marine data and will be skipped.")
marine_data_not_gt = marine_data.loc[covered_not_gt_dt]
samples = marine_data_not_gt[marine_data_not_gt["tide_level_quantile"].isin([2, 3])].sample(10)
print(f"Random samples of 3th and 4th quantiles of the tide level distribution that are not included in GT - {STUDY_NAME}:")
print(samples[["tide", "tide_is_rising", "tide_level_quantile"]])

## EXTRA: Show the marine data for specific date:
# date_ex = "2016-06-19 11:00:00"
date_ex = "2016-06-26 11:00:00"

print(f"\n\nMarine data for {date_ex} - {STUDY_NAME}:")
print(marine_data.loc[date_ex])

# Count how many images in the RGB TIFF directory have less than date_ex values for hs and tide level
count_hs = 0
count_tide = 0
for file in os.listdir(RGB_TIFF_DIR):
    if file.endswith(".png"):
        dt = _parse_dt_to_hour_res(file)
        if dt in marine_data.index:
            if marine_data.loc[dt, "hs"] < marine_data.loc[date_ex, "hs"]:
                count_hs += 1
            if marine_data.loc[dt, "tide"] < marine_data.loc[date_ex, "tide"]:
                count_tide += 1

print(f"\n\nCount of images with less than {date_ex} values for hs and tide level - {STUDY_NAME}:")
print(f"hs: {count_hs}")
print(f"tide: {count_tide}")