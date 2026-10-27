import numpy as np
from pathlib import Path
import pandas as pd

def check_shapes(metadata_path: str):
    base_dir = Path(metadata_path).parent
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"File {metadata_path} not found.")
        return

    print(f"Checking {len(df)} records in {metadata_path}...")
    errors = 0
    for idx, row in df.iterrows():
        img_path = base_dir / row["image_npy_path"]
        mask_path = base_dir / row["mask_npy_path"]
        
        try:
            img = np.load(img_path)
            mask = np.load(mask_path)
            
            if img.shape != (6, 128, 128):
                print(f"Error: Invalid image shape {img.shape} for {img_path.name}")
                errors += 1
            if mask.shape != (128, 128):
                print(f"Error: Invalid mask shape {mask.shape} for {mask_path.name}")
                errors += 1
                
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
            errors += 1
            
    if errors == 0:
        print("Success! All patches are exactly (6, 128, 128) and (128, 128).")
    else:
        print(f"Finished with {errors} errors.")

if __name__ == "__main__":
    train_csv = "data/dl_npy/train/metadata.csv"
    test_csv = "data/dl_npy/test/metadata.csv"
    
    print("\n--- TRAIN SPLIT ---")
    check_shapes(train_csv)
    
    print("\n--- TEST SPLIT ---")
    check_shapes(test_csv)
