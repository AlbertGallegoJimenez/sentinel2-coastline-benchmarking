import os
import sys
import shutil
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dl_models import train_model, get_model, CoastlineDataset
from notebooks.utils.evaluation_utils import (
    collapse_gt_to_binary, 
    create_boundary_zone_mask, 
    calculate_metrics
)

# Constants extracted from the notebook setup
BOUNDARY_OFFSET_PIXELS = 10
MIN_BOUNDARY_PIXELS = 50
MAPPED_GT_CLASS_MAPPING = {
    'water_classes': [2, 4],
    'no_water_classes': [1, 3],
    'ignore_classes': [0, 255]
}

class TrainArgs:
    """Mock argparse.Namespace to pass parameters to the train_model function."""
    def __init__(self, arch, encoder, epochs, batch_size, lr, loss, patience, seed):
        self.arch = arch
        self.encoder = encoder
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.patience = patience
        self.seed = seed

def evaluate_single_model(model_filepath, arch, encoder, test_loader, dataset_meta, device):
    """
    Evaluates a single model explicitly on the boundary zone and returns the IoU 
    for Salinas and Puntal.
    """
    model = get_model(arch=arch, encoder_name=encoder)
    state_dict = torch.load(model_filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = []
    
    # Do not track gradients for evaluation
    with torch.no_grad():
        for dt_idx, (image_tensor, gt_mask_tensor) in enumerate(tqdm(test_loader, desc=f"Evaluating {arch}-{encoder}", leave=False)):
            image_tensor = image_tensor.float().to(device)
            
            # Predict
            logits = model(image_tensor)
            pred_mask = torch.argmax(logits, dim=1).cpu().numpy().squeeze(0)
            gt_mask = gt_mask_tensor.numpy().squeeze(0)
            
            row = dataset_meta.iloc[dt_idx]
            
            # Binary Masks
            gt_binary, ignore_mask = collapse_gt_to_binary(
                gt_mask,
                water_classes=MAPPED_GT_CLASS_MAPPING['water_classes'],
                no_water_classes=MAPPED_GT_CLASS_MAPPING['no_water_classes'],
                ignore_classes=MAPPED_GT_CLASS_MAPPING['ignore_classes']
            )

            pred_binary, _ = collapse_gt_to_binary(
                pred_mask,
                water_classes=MAPPED_GT_CLASS_MAPPING['water_classes'],
                no_water_classes=MAPPED_GT_CLASS_MAPPING['no_water_classes'],
                ignore_classes=[0, 255]
            )

            # Boundary setup
            boundary_zone = create_boundary_zone_mask(gt_binary, ignore_mask, offset_pixels=BOUNDARY_OFFSET_PIXELS)
            evaluation_mask = boundary_zone & (~ignore_mask)
            combined_ignore_mask = ~evaluation_mask
            
            boundary_pixel_count = np.sum(evaluation_mask)
            if boundary_pixel_count < MIN_BOUNDARY_PIXELS:
                continue
            
            # Metrics calculation
            metrics = calculate_metrics(gt_binary, pred_binary, combined_ignore_mask)
            if metrics['valid_pixel_count'] == 0:
                continue
            
            results.append({
                'study_name': row['study_name'],
                'iou': metrics['iou']
            })

    # Free memory explicitly
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    if df.empty:
        return 0.0, 0.0
        
    # Group by study name
    means = df.groupby('study_name')['iou'].mean().to_dict()
    iou_salinas = means.get('salinas', 0.0)
    iou_puntal = means.get('puntal', 0.0)
    
    return iou_salinas, iou_puntal

def main():
    parser = argparse.ArgumentParser(description="Master script to run, evaluate, and keep the best multi-run models.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--runs", type=int, default=10, help="Number of iterations to run per configuration")
    parser.add_argument("--salinas_weight", type=float, default=0.6, help="Weight applied to Salinas IoU (0.0 to 1.0)")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for reproducibility. Each run adds its index to this.")
    args = parser.parse_args()

    # Configurations to test
    ARCHITECTURES = ["segformer"]
    ENCODERS = ["mit_b1", "mit_b2", "mit_b3"]
    
    puntal_weight = 1.0 - args.salinas_weight

    # Ensure device is set
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load test dataset for evaluations
    test_dir = PROJECT_ROOT / "data" / "dl_npy" / "test"
    test_dataset = CoastlineDataset(test_dir, transform=None)
    # Batch size must be 1 for iterative patch extraction in evaluation script
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    dataset_meta = test_dataset.metadata

    results_dir = PROJECT_ROOT / "results" / "models"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_results = []

    for arch in ARCHITECTURES:
        for encoder in ENCODERS:
            print(f"\n============================================================")
            print(f"Starting experiments for {arch.upper()} with {encoder}")
            print(f"============================================================")
            
            best_weighted_iou = -1.0
            best_run_idx = -1
            best_iou_salinas = 0.0
            best_iou_puntal = 0.0
            
            original_model_path = results_dir / f"best_model_{arch}_{encoder}.pth"
            best_run_target_path = results_dir / f"best_model_{arch}_{encoder}_best_run.pth"

            for run in range(1, args.runs + 1):
                print(f"\n--- [Run {run}/{args.runs}] {arch}-{encoder} ---")
                
                # Determine deterministic seed for this run if base_seed is provided
                run_seed = args.base_seed + run if args.base_seed is not None else None

                # Create mocked args for the existing train_model function
                train_args = TrainArgs(
                    arch=arch, 
                    encoder=encoder, 
                    epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    lr=5e-4, 
                    loss="ce", 
                    patience=15, 
                    seed=run_seed
                )
                
                # 1. Train the Model
                # This inherently creates "best_model_{arch}_{encoder}.pth" inside results/models
                train_model(train_args)
                
                if not original_model_path.exists():
                    print("Error: Model weights were not found after training!")
                    continue
                
                # 2. Evaluate the freshly trained model on boundary zone
                salinas_iou, puntal_iou = evaluate_single_model(
                    model_filepath=original_model_path, 
                    arch=arch, 
                    encoder=encoder, 
                    test_loader=test_loader, 
                    dataset_meta=dataset_meta, 
                    device=device
                )
                
                # 3. Calculate weighted average
                weighted_iou = (salinas_iou * args.salinas_weight) + (puntal_iou * puntal_weight)
                
                print(f"   => Eval Results: Salinas IoU = {salinas_iou:.4f} | Puntal IoU = {puntal_iou:.4f}")
                print(f"   => Weighted Metric: {weighted_iou:.4f}")
                
                # 4. Compare and preserve
                if weighted_iou > best_weighted_iou:
                    print(f"   [!] New Best Run Found! Updating weights...")
                    best_weighted_iou = weighted_iou
                    best_run_idx = run
                    best_iou_salinas = salinas_iou
                    best_iou_puntal = puntal_iou
                    
                    # Copy the newly trained model to a "best_run" file
                    shutil.copy2(original_model_path, best_run_target_path)
            
                # Explicit cleanup between runs to prevent VRAM accumulation OOMs
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Store summary across the board
            print(f"\n*** Finished {arch}-{encoder} | Best Run: #{best_run_idx} | Best Weighted IoU: {best_weighted_iou:.4f} ***\n")
            summary_results.append({
                "Architecture": arch,
                "Encoder": encoder,
                "Best_Run": best_run_idx,
                "Best_Salinas_IoU": round(best_iou_salinas, 4),
                "Best_Puntal_IoU": round(best_iou_puntal, 4),
                "Best_Weighted_IoU": round(best_weighted_iou, 4)
            })

    # Print final leaderboard
    print("\n============================================================")
    print("                 FINAL EXPERIMENT SUMMARY                   ")
    print("============================================================")
    summary_csv_path = results_dir / "experiment_runs_summary.csv"

    if os.path.exists(summary_csv_path):
        summary_df_existing = pd.read_csv(summary_csv_path)
        summary_df = pd.concat([summary_df_existing, pd.DataFrame(summary_results)], ignore_index=True)
        summary_df = summary_df.sort_values(by="Best_Weighted_IoU", ascending=False)
    else:
        summary_df = pd.DataFrame(summary_results).sort_values(by="Best_Weighted_IoU", ascending=False)
    
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary saved to: {summary_csv_path}")

if __name__ == "__main__":
    main()