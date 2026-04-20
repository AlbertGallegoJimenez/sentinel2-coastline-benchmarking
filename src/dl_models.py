import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
from tqdm import tqdm

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

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        
        # Mask for CrossEntropyLoss should be Long tensor of shape (H, W) without channel dim
        mask = torch.tensor(mask_mapped, dtype=torch.long)

        if self.transform:
            # Albumentations expects HWC if we use it later, 
            # but for now we'll pass. We can implement on-the-fly here.
            pass
            
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


def train_prototype():
    """
    Training loop designed for prototyping on Apple Silicon (MPS).
    """
    # 1. Setup parameters and device
    epochs = 3
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
    
    if not train_dir.exists():
        print(f"Error: Directory {train_dir} not found.")
        return

    print("Loading dataset and dataloader...")
    train_dataset = CoastlineDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Train dataset loaded: {len(train_dataset)} patches.")

    # 3. Initialize Model, Optimizer and Loss
    print("Instantiating U-Net (ResNet34) with 6 in_channels and 5 classes...")
    model = get_model(arch="unet", encoder_name="resnet34").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Multi-class segmentation loss with ignore_index for the 255 value (no-data/ignore)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    # 4. Training Loop (Prototype)
    print("\nStarting training loop (prototype)...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Progress bar for each epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, masks) in enumerate(progress_bar):
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
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"--> [Epoch {epoch+1}] Average loss: {avg_loss:.4f}\n")
        
    print("Prototype finished successfully! The pipeline runs properly.")

if __name__ == "__main__":
    train_prototype()