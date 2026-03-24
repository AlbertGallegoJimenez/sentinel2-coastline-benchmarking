import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from rasterio.transform import xy

# Set data paths
STUDY_NAME = "salinas"
MS_DIR = f"data/images/{STUDY_NAME}/ms"
SWIR_DIR = f"data/images/{STUDY_NAME}/swir"

def get_tif_filename(date_str, im_type="ms"):
    if im_type == "ms":
        filename = [f for f in os.listdir(MS_DIR) if f.startswith(date_str) and f.endswith(".tif")][0]
    else:
        filename = [f for f in os.listdir(SWIR_DIR) if f.startswith(date_str) and f.endswith(".tif")][0]
    return filename

def get_band_data(ms_path, swir_path):
    with rasterio.open(ms_path) as ms_src:
        blue = ms_src.read(1).astype(float)
        green = ms_src.read(2).astype(float)
        red = ms_src.read(3).astype(float)
        nir = ms_src.read(4).astype(float)
        transform = ms_src.transform
        profile = ms_src.profile

    with rasterio.open(swir_path) as swir_src:
        swir1 = swir_src.read(1).astype(float)
        
    ndwi = (green - nir) / (green + nir + 1e-10)
    mndwi = (green - swir1) / (green + swir1 + 1e-10)

    # Clean boundaries due to SWIR resolution artifacts
    mndwi[:2, :] = np.nan
    mndwi[:, :4] = np.nan
    mndwi[:, -3:] = np.nan

    return {"Blue": blue, "Green": green, "Red": red, "NIR": nir, "SWIR1": swir1, "NDWI": ndwi, "MNDWI": mndwi}, transform, profile

# Target date
target_date = "2016-11-16"

ms_filename = get_tif_filename(target_date, "ms")
swir_filename = get_tif_filename(target_date, "swir")

ms_path = os.path.join(MS_DIR, ms_filename)
swir_path = os.path.join(SWIR_DIR, swir_filename)

bands_data, transform, profile = get_band_data(ms_path, swir_path)

# Calculate Otsu thresholds on valid data
valid_ndwi = bands_data["NDWI"][~np.isnan(bands_data["NDWI"])]
otsu_ndwi = threshold_otsu(valid_ndwi)
valid_mndwi = bands_data["MNDWI"][~np.isnan(bands_data["MNDWI"])]
otsu_mndwi = threshold_otsu(valid_mndwi)

print(f"Otsu NDWI: {otsu_ndwi:.4f}, Otsu MNDWI: {otsu_mndwi:.4f}")

# Extract contours and save them
def save_contour(image_data, threshold, transform, crs, output_path):
    contours = find_contours(image_data, threshold)
    lines = []
    for contour in contours:
        # contour is array of (row, col)
        coords = [xy(transform, r, c) for r, c in contour]
        if len(coords) > 1:
            lines.append(LineString(coords))
            
    if lines:
        gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=crs)
        gdf_lines.to_file(output_path)
        print(f"Saved contour to {output_path}")

output_dir = "results/contours"
os.makedirs(output_dir, exist_ok=True)
save_contour(bands_data["NDWI"], otsu_ndwi, transform, profile['crs'], os.path.join(output_dir, f"NDWI_contour_{target_date}.shp"))
save_contour(bands_data["MNDWI"], otsu_mndwi, transform, profile['crs'], os.path.join(output_dir, f"MNDWI_contour_{target_date}.shp"))

# Load transect
transect_path = r"C:\Users\usuari\Desktop\TFM\test_s2\test_s2\test_s2.gdb"
layer_name = "example_cross_section"
gdf = gpd.read_file(transect_path, layer=layer_name)

# Assume there is only one line in the geodataframe
line = gdf.geometry.iloc[0]

# Load actual waterline
gdf_waterline = gpd.read_file(transect_path, layer="actual_waterline_example")
actual_waterline = gdf_waterline.geometry.iloc[0]

# Calculate intersection point and corresponding cross-shore distance
intersection = line.intersection(actual_waterline)
if not intersection.is_empty:
    dist_on_line = line.project(intersection)
    max_d = np.arange(0, line.length, 10)[-1]
    # Mapping to plot's X axis (which is max_d - dist_on_line based on points reversal)
    actual_waterline_dist = max_d - dist_on_line
else:
    actual_waterline_dist = None

# Interpolate points every 10 meters
distance = np.arange(0, line.length, 10)
points = [line.interpolate(dist) for dist in distance][::-1]  # Reverse to go from sea to land

# Extract values for each band at these points
band_values = {band: [] for band in bands_data.keys()}

with rasterio.open(ms_path) as src:
    for point in points:
        # Convert coords to pixel row, col
        row, col = src.index(point.x, point.y)
        for band_name, band_array in bands_data.items():
            if 0 <= row < band_array.shape[0] and 0 <= col < band_array.shape[1]:
                band_values[band_name].append(band_array[row, col])
            else:
                band_values[band_name].append(np.nan)

# Calculate cross-shore intersections for NDWI and MNDWI with Otsu thresholds
cross_ndwi, cross_mndwi = None, None
ndwi_vals = np.array(band_values["NDWI"])
mndwi_vals = np.array(band_values["MNDWI"])
dist_array = np.array(distance)

for i in range(len(dist_array) - 1):
    if not np.isnan(ndwi_vals[i]) and not np.isnan(ndwi_vals[i+1]):
        if (ndwi_vals[i] - otsu_ndwi) * (ndwi_vals[i+1] - otsu_ndwi) <= 0 and cross_ndwi is None:
            # Interpolate exact distance
            cross_ndwi = dist_array[i] + (otsu_ndwi - ndwi_vals[i]) * (dist_array[i+1] - dist_array[i]) / (ndwi_vals[i+1] - ndwi_vals[i])
            
    if not np.isnan(mndwi_vals[i]) and not np.isnan(mndwi_vals[i+1]):
        if (mndwi_vals[i] - otsu_mndwi) * (mndwi_vals[i+1] - otsu_mndwi) <= 0 and cross_mndwi is None:
            cross_mndwi = dist_array[i] + (otsu_mndwi - mndwi_vals[i]) * (dist_array[i+1] - dist_array[i]) / (mndwi_vals[i+1] - mndwi_vals[i])

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
colors = {"Blue": "blue", "Green": "green", "Red": "red", "NIR": "purple", "SWIR1": "orange", "NDWI": "cyan", "MNDWI": "magenta"}

for band_name, values in band_values.items():
    if band_name in ["NDWI", "MNDWI"]:
        continue
    plt.plot(distance, values, label=band_name, color=colors.get(band_name, "black"), lw=1)

# Add axvline
# if cross_ndwi is not None:
#     plt.axvline(x=cross_ndwi, color='cyan', linestyle='--', label=f'NDWI Otsu (d={cross_ndwi:.1f}m)')
if actual_waterline_dist is not None:
    plt.axvline(x=actual_waterline_dist, color='black', lw=2, linestyle='-', label='Actual Waterline')

if cross_mndwi is not None:
    plt.axvline(x=cross_mndwi, color='k', lw=2, linestyle='--', label=f"MNDWI Otsu's Waterline")

# Clean up the layout
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.4)

plt.xlabel("Cross-shore distance (m)")
plt.ylabel("Intensity")
#plt.title(f"Cross-section of Sentinel-2 bands for {target_date}")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
fig.savefig("example_cross_section.png", dpi=300, bbox_inches='tight')
