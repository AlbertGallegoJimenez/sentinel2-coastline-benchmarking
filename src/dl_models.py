import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A

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

def train_prototype():
    """
    Training loop designed for prototyping on Apple Silicon (MPS).
    """
    # 1. Setup parameters and device
    epochs = 5
    batch_size = 16
    lr = 1e-4
    
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
    architecture = "unet"
    encoder = "resnet34"
    
    print(f"Instantiating {architecture.upper()} ({encoder}) with 6 in_channels and 5 classes...")
    model = get_model(arch=architecture, encoder_name=encoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Multi-class segmentation loss with ignore_index for the 255 value (no-data/ignore)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    # 4. Training Loop (Prototype)
    print("\nStarting training loop (prototype)...")
    
    best_val_loss = float('inf')
    
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
        
        print(f"--> [Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = Path("results/models")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = f"best_model_{architecture}_{encoder}.pth"
            torch.save(model.state_dict(), output_dir / model_filename)
            print(f"    [!] New best model saved as {model_filename} with Val Loss: {best_val_loss:.4f}")
        print()
        
    print("Prototype finished successfully! The pipeline runs properly.")

if __name__ == "__main__":
    train_prototype()