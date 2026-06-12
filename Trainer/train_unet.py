import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

class BreakHisSegmentationDataset(Dataset):
    def __init__(self, data_dir, metadata_file, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter for images that have segmentation masks (if available)
        self.images = []
        for item in self.metadata:
            if 'image_path' in item:
                self.images.append(item)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        image_path = os.path.join(self.data_dir, item['image_path'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # For now, create a simple binary mask based on image intensity
        # In a real scenario, you would have actual segmentation annotations
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            gray = np.mean(image_np, axis=2)
        else:
            gray = image_np
        
        # Create a simple mask based on intensity thresholding
        # This is a placeholder - replace with real annotations
        _, mask = cv2.threshold(gray.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # If no target transform, convert to tensor manually
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def train_unet():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        # Ensure values are in [0,1] range
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    
    # Create dataset and dataloader
    data_dir = "data/breakhis_organized"
    metadata_file = "data/breakhis_training_metadata.json"
    
    # Always create fresh metadata for training
    print("Creating metadata file for training...")
    create_simple_metadata(data_dir, metadata_file)
    
    dataset = BreakHisSegmentationDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        transform=transform,
        target_transform=target_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)}")
    
    # Create U-Net model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid'
    )
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Save model every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'models/unet_segmentation_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'models/unet_segmentation.pth')
    print("Training completed! Model saved to models/unet_segmentation.pth")

def create_simple_metadata(data_dir, metadata_file):
    """Create a simple metadata file for training"""
    metadata = []
    
    # Check for numbered folders (0 for benign, 1 for malignant)
    class_folders = ['0', '1']  # 0=benign, 1=malignant
    
    for class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.exists(class_path):
            class_label = 'benign' if class_idx == 0 else 'malignant'
            
            # Get all image files in this class folder
            for file in os.listdir(class_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.join(class_folder, file)
                    
                    metadata.append({
                        'image_path': rel_path,
                        'class': class_label,
                        'filename': file
                    })
    
    # Save metadata
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata file with {len(metadata)} images")
    print(f"Classes found: {set([item['class'] for item in metadata])}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    train_unet()
