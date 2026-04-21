import os
import psutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
import argparse
import random

def set_seed(seed=42):
    """
    Freeze all randomness for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CoastlineDataset(Dataset):
    """
    Custom Dataset for Sentinel-2 coastline extraction.
    Loads 6-channel .npy images and binary masks.
    """
    def __init__(self, split_dir, transform=None):
        self.split_dir = Path(split_dir)
        self.metadata = pd.read_csv(self.split_dir / "metadata.csv")
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = self.split_dir / row["image_npy_path"]
        mask_path = self.split_dir / row["mask_npy_path"]

        # Load numpy arrays
        # Image expected shape from preprocessing: (6, 128, 128) -> (C, H, W)
        image = np.load(image_path)
        # Mask expected shape: (128, 128)
        mask = np.load(mask_path)

        # Map mask values from 1-5 to 0-4 (PyTorch CrossEntropy expects 0-indexed classes)
        # Keep 255 as ignore_index
        mask_mapped = np.where(mask == 255, 255, mask - 1)

        if self.transform:
            # Albumentations expects images in HWC format (Height, Width, Channels)
            # Currently image is CHW: (6, 128, 128)
            image_hwc = np.transpose(image, (1, 2, 0))
            
            augmented = self.transform(image=image_hwc, mask=mask_mapped)
            image_hwc = augmented['image']
            mask_mapped = augmented['mask']
            
            # Convert back to CHW for PyTorch
            image = np.transpose(image_hwc, (2, 0, 1))

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        
        # Mask for CrossEntropyLoss should be Long tensor of shape (H, W) without channel dim
        mask = torch.tensor(mask_mapped, dtype=torch.long)

        return image, mask


def get_model(arch="unet", encoder_name="resnet34", encoder_weights="imagenet"):
    """
    Instantiate a Segmentation Model modifying the first layer to accept 6 channels.
    """
    # 5 Classes: clouds, sand, water, wet_sand, white_water
    global num_classes  # Let the loss function access it below
    num_classes = 5

    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=6,        # RGB (3) + NIR (1) + SWIR (2)
            classes=num_classes,  # Multi-class prediction
        )
    elif arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=6,
            classes=num_classes,
        )
    elif arch == "segformer":
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=6,
            classes=num_classes,
        ) 
    else:
        raise ValueError(f"Architecture '{arch}' not supported in this prototype.")
        
    return model


def get_training_augmentations():
    """
    Return training data augmentations using Albumentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
    ])

def train_model(args):
    """
    Full training loop for DL models on Coastline extraction.
    """
    # Fix the random seed if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}...")
        set_seed(args.seed)

    # 1. Setup parameters from args and device
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    architecture = args.arch
    encoder = args.encoder
    
    # Check for Apple Silicon backend (MPS) or CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS backend detected. Using device:", device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA backend detected. Using device:", device)
    else:
        device = torch.device("cpu")
        print("MPS and CUDA not found. Using CPU.")

    # 2. Load Dataloaders
    base_data_path = Path("data/dl_npy")
    train_dir = base_data_path / "train"
    test_dir = base_data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"Error: Directories not found in {base_data_path}.")
        return

    print("Loading datasets and dataloaders...")
    
    train_transforms = get_training_augmentations()
    train_dataset = CoastlineDataset(train_dir, transform=train_transforms)
    # Validation/Test dataset doesn't use data augmentation
    test_dataset = CoastlineDataset(test_dir, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Train dataset loaded: {len(train_dataset)} patches.")
    print(f"Test dataset loaded: {len(test_dataset)} patches.")

    # 3. Initialize Model, Optimizer and Loss
    print(f"Instantiating {architecture.upper()} ({encoder}) with 6 in_channels and 5 classes...")
    model = get_model(arch=architecture, encoder_name=encoder).to(device)
    
    # [TWEAK 1] Optimizer: Changed from Adam to AdamW (adds weight decay for better regularisation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning Rate Scheduler: reduces LR if validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # [TWEAK 2] Loss Function selection (vital for class imbalance in coastline pixels)
    if args.loss == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    elif args.loss == "focal":
        loss_fn = smp.losses.FocalLoss(mode='multivalue' if num_classes==1 else 'multiclass', ignore_index=255)
    elif args.loss == "dice":
        loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    # early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0

    # 4. Training Loop
    print(f"\nStarting full training loop for {epochs} epochs...")
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        # Progress bar for each epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in progress_bar:
            # Move data to MPS/GPU
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss and backward pass
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            # Update tracking
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for images, masks in val_progress:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(test_loader)
        
        # --- RESOURCE TRACKING ---
        import psutil # ensure imported here if not at top
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        mem_str = f"CPU: {cpu_usage}% | System RAM: {ram_usage}%"
        if device.type == "mps":
            mps_alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
            mem_str += f" | MPS VRAM Alloc: {mps_alloc:.1f} MB"
        elif device.type == "cuda":
            cuda_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_str += f" | CUDA VRAM Alloc: {cuda_alloc:.1f} MB"
            
        print(f"--> [Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    [Hardware] {mem_str}")
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model and Early Stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            output_dir = Path("results/models")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = f"best_model_{architecture}_{encoder}.pth"
            torch.save(model.state_dict(), output_dir / model_filename)
            print(f"    [!] New best model saved as {model_filename} with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    [!] No improvement. Early stopping counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping triggered after {epoch+1} epochs.")
                break
        print()
        
    print(f"Training finished! Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DL models for Coastline Extraction on Sentinel-2.")
    parser.add_argument("--arch", type=str, default="unet", help="Architecture (unet, deeplabv3plus, segformer)")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder backbone (e.g., resnet34, mit_b0)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoaders")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--loss", type=str, default="ce", help="Loss function: ce (CrossEntropy), focal, or dice.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic training. Leave empty for random.")
    
    args = parser.parse_args()
    
    train_model(args)