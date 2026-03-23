import os
import glob
import math
import geopandas as gpd
from shapely.geometry import box

# Configuración de variables
BASE_DIR = r"C:\Users\usuari\Documents\repos github\sentinel2-coastline-benchmarking\data\coastal_rois_for_patches"
PIXEL_RESOLUTION = 10  # Considerando 10 metros de resolución espacial (típico Sentinel-2)
PATCH_SIZE_PIXELS = 128 # Tamaño de los parches (256x256)
OVERLAP_PERCENT = 0.15 # 15% de solape
MIN_AREA_RATIO = 0.005  # Si más del 50% de los píxeles caen fuera, se descarta

# Variables escaladas en unidades del mundo real (metros) si el CRS del shapefile es proyectado
PATCH_SIZE_M = PATCH_SIZE_PIXELS * PIXEL_RESOLUTION

# Cada parche solapa un 15% del anterior a lo largo de un eje. 
# El avance es el parche menos la cantidad a solapar.
overlap_m = PATCH_SIZE_M * OVERLAP_PERCENT
STRIDE_M = PATCH_SIZE_M - overlap_m

def load_rois_and_process():
    shp_files = glob.glob(os.path.join(BASE_DIR, "**", "*.shp"), recursive=True)
    
    for shp_path in shp_files:
        # Evitar procesar los shapefiles de parches ya creados si los pisa
        if '_patches.shp' in shp_path:
            continue
            
        print(f"-> Procesando: {shp_path}")
        try:
            gdf = gpd.read_file(shp_path)
            
            # Reproyectar a un SRC métrico (ej EPSG 3857, o ya estarán en UTM). 
            # Aquí asumimos que los datos ya están en metros (coord. UTM) para resolucion de 10m
            if gdf.crs and gdf.crs.is_geographic:
                print("Advertencia: El CRS es geográfico (grados). Convierte a un CRS proyectado (metros) para un cálculo preciso.")
                # gdf = gdf.to_crs(epsg=32630) # Ejemplo UTM 30N
        except Exception as e:
            print(f"Error cargando {shp_path}: {e}")
            continue

        all_patches = []
        
        # Iterar sobre cada polígono/ROI dentro del archivo (ej. puntal tiene 2 ROIs)
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
                
            minx, miny, maxx, maxy = geom.bounds
            
            # Generar el grid sobre el Bounding Box del polígono actual
            x_start = minx
            while x_start + PATCH_SIZE_M <= maxx:
                y_start = miny
                while y_start + PATCH_SIZE_M <= maxy:
                    
                    # Generar parche 256x256 pix
                    patch_geom = box(x_start, y_start, x_start + PATCH_SIZE_M, y_start + PATCH_SIZE_M)
                    
                    # Comprobar la intersección con el ROI original
                    intersection_area = geom.intersection(patch_geom).area
                    
                    # Evaluar si la proporción guardada en el ROI supera el 50%
                    if intersection_area / patch_geom.area >= MIN_AREA_RATIO:
                        all_patches.append(patch_geom)
                    
                    # Avanzar usando el stride
                    y_start += STRIDE_M
                
                # Avanzar eje X
                x_start += STRIDE_M
                
        if all_patches:
            # Crear GeoDataFrame para exportar los parches válidos detectados
            patches_gdf = gpd.GeoDataFrame({'geometry': all_patches}, crs=gdf.crs)
            out_patch = f'_{PATCH_SIZE_PIXELS}_{PATCH_SIZE_PIXELS}_patches'
            output_name = shp_path.replace('.shp', f'{out_patch}.shp')
            
            # Guardar a disco
            #patches_gdf.to_file(output_name)
            print(f"   Creados {len(all_patches)} parches en: {output_name}")
        else:
            print("   Ningún parche cumplió con el solape requerido > 50%.")

if __name__ == '__main__':
    load_rois_and_process()
