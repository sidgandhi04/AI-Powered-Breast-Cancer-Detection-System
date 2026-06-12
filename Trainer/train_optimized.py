import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import os
import json
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for research-quality plots
plt.style.use('default')
sns.set_palette("husl")

class BreakHisClassificationDataset(Dataset):
    def __init__(self, data_dir, metadata_file, transform=None, split='train', train_ratio=0.7, val_ratio=0.15):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        np.random.seed(42)
        np.random.shuffle(self.metadata)
        
        n_total = len(self.metadata)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        if split == 'train':
            self.images = self.metadata[:n_train]
        elif split == 'val':
            self.images = self.metadata[n_train:n_train + n_val]
        else:
            self.images = self.metadata[n_train + n_val:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        image_path = os.path.join(self.data_dir, item['image_path'])
        image = Image.open(image_path).convert('RGB')
        label = 0 if item['class'] == 'benign' else 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(architecture, num_classes=2):
    """Create different CNN models"""
    if architecture == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif architecture == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

def monitor_gpu_performance():
    """Monitor GPU performance during training"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"📊 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.1f}GB total")
        
        # Try to get GPU utilization, handle errors gracefully
        try:
            utilization = torch.cuda.utilization(0)
            print(f"📊 GPU Utilization: {utilization}%")
        except Exception as e:
            print(f"📊 GPU Utilization: Unable to read (CUDA driver limitation)")
        
        return memory_allocated, memory_reserved, memory_total
    return 0, 0, 0

def train_cnn_model(model, train_loader, val_loader, device, num_epochs=8, model_name="model"):
    """Train a CNN model with validation - OPTIMIZED FOR CUDA"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Enable mixed precision training for faster CUDA performance
    scaler = torch.cuda.amp.GradScaler()
    
    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"🚀 Training {model_name} with Mixed Precision + CUDA Optimizations")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}')):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Clear GPU cache periodically and monitor performance
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            monitor_gpu_performance()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def plot_training_results(cnn_models, save_dir='results'):
    """Plot training results for research paper"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimized CNN Training Results for Breast Cancer Detection', fontsize=18, fontweight='bold')
    
    # Training Loss
    ax1 = axes[0, 0]
    for name, data in cnn_models.items():
        ax1.plot(data['train_losses'], label=name, linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2 = axes[0, 1]
    for name, data in cnn_models.items():
        ax2.plot(data['val_losses'], label=name, linewidth=2, alpha=0.8)
    ax2.set_title('Validation Loss Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3 = axes[1, 0]
    for name, data in cnn_models.items():
        ax3.plot(data['train_accs'], label=name, linewidth=2, alpha=0.8)
    ax3.set_title('Training Accuracy Evolution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final Validation Accuracy Comparison
    ax4 = axes[1, 1]
    names = list(cnn_models.keys())
    val_accs = [cnn_models[name]['best_val_acc'] for name in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax4.bar(names, val_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_title('Final Validation Accuracy by Model', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Model Architecture', fontsize=12)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, val_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimized_training_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training results visualization saved!")

def main():
    """Main training function - OPTIMIZED VERSION"""
    print("🚀 Starting OPTIMIZED Model Training for Breast Cancer Detection")
    print("=" * 70)
    
    # Set device and optimize CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # CUDA Performance Optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print("🚀 CUDA Optimizations Enabled:")
        print("   - cudnn.benchmark: True")
        print("   - TF32: Enabled")
        print("   - Memory fraction: 90%")
        
        # Clear any existing cache
        torch.cuda.empty_cache()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    data_dir = "data/breakhis_organized"
    metadata_file = "data/breakhis_training_metadata.json"
    
    if not os.path.exists(metadata_file):
        from train_unet import create_simple_metadata
        print("Creating metadata file...")
        create_simple_metadata(data_dir, metadata_file)
    
    train_dataset = BreakHisClassificationDataset(data_dir, metadata_file, transform, 'train')
    val_dataset = BreakHisClassificationDataset(data_dir, metadata_file, transform, 'val')
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Total images: {len(train_dataset) + len(val_dataset)}")
    print(f"Training on entire dataset with optimized batch size")
    
    # Create dataloaders - optimized for CUDA performance (Windows compatible)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory >= 8:
        optimal_batch_size = 64
        num_workers = 0  # Windows compatibility
    elif gpu_memory >= 6:
        optimal_batch_size = 48
        num_workers = 0  # Windows compatibility
    else:
        optimal_batch_size = 32
        num_workers = 0  # Windows compatibility
    
    print(f"🚀 GPU Memory: {gpu_memory:.1f}GB → Optimal Batch Size: {optimal_batch_size}, Workers: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    # Focus on key models for faster training
    cnn_architectures = ['resnet18', 'resnet34', 'densenet121', 'mobilenet_v2', 'efficientnet_b0']
    
    # Train CNN models
    print("\n🔬 Training Key CNN Models (Optimized)...")
    print("=" * 50)
    
    cnn_models = {}
    total_start_time = time.time()
    
    for arch in cnn_architectures:
        try:
            print(f"\n{'='*20} Training {arch.upper()} {'='*20}")
            model_start_time = time.time()
            
            model = create_model(arch)
            results = train_cnn_model(model, train_loader, val_loader, device, num_epochs=8, model_name=arch.upper())
            cnn_models[arch.upper()] = results
            
            # Save model
            os.makedirs('models', exist_ok=True)
            torch.save(results['model'].state_dict(), f'models/{arch}_breast_cancer_optimized.pth')
            
            model_time = time.time() - model_start_time
            print(f"✅ {arch.upper()} trained and saved in {model_time:.1f} seconds!")
            print(f"🏆 Best Validation Accuracy: {results['best_val_acc']:.2f}%")
            
        except Exception as e:
            print(f"❌ Error training {arch}: {str(e)}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Generate results
    print("\n📊 Generating Research-Quality Results...")
    print("=" * 50)
    
    os.makedirs('results', exist_ok=True)
    plot_training_results(cnn_models, 'results')
    
    # Performance summary
    print("\n🎉 OPTIMIZED Training Complete!")
    print("=" * 40)
    print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes")
    print(f"🚀 Models Trained: {len(cnn_models)}")
    print(f"📊 Best Model: {max(cnn_models.items(), key=lambda x: x[1]['best_val_acc'])[0]}")
    print(f"🏆 Best Accuracy: {max(cnn_models.items(), key=lambda x: x[1]['best_val_acc'])[1]['best_val_acc']:.2f}%")
    
    # Save performance summary
    performance_data = []
    for name, data in cnn_models.items():
        performance_data.append({
            'Model': name,
            'Best_Val_Accuracy': data['best_val_acc'],
            'Final_Train_Loss': data['train_losses'][-1],
            'Final_Val_Loss': data['val_losses'][-1]
        })
    
    df = pd.DataFrame(performance_data)
    df = df.sort_values('Best_Val_Accuracy', ascending=False)
    df.to_csv('results/optimized_performance_summary.csv', index=False)
    
    print(f"📁 Results saved in 'results/' directory")
    print(f"🤖 Models saved in 'models/' directory")
    print(f"📊 Performance summary saved as 'results/optimized_performance_summary.csv'")

if __name__ == "__main__":
    main()
