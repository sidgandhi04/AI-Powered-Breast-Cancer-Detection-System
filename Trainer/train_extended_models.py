import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class
class BreakHisExtendedDataset(Dataset):
    def __init__(self, metadata_file, transform=None, target_transform=None):
        with open(metadata_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

# Model creation function for extended models
def create_extended_model(model_name, num_classes=2):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet169':
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    elif model_name == 'shufflenet_v2_x1_5':
        model = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# Training function
def train_extended_model(model_name, train_loader, val_loader, num_epochs=8, learning_rate=0.001):
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create model
    model = create_extended_model(model_name)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_path = f'models/{model_name}_breast_cancer_extended.pth'
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

# Evaluation function
def evaluate_extended_model(model, test_loader, model_name):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Main training function
def main():
    print("🚀 Starting Extended Model Training (10 New Models)")
    print("=" * 60)
    
    # Check if metadata exists
    metadata_file = 'data/breakhis_training_metadata.json'
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        print("Please run train_unet.py first to generate the metadata")
        return
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Loaded {len(metadata)} images from metadata")
    
    # Split data (80% train, 10% val, 10% test)
    np.random.shuffle(metadata)
    split_idx1 = int(0.8 * len(metadata))
    split_idx2 = int(0.9 * len(metadata))
    
    train_data = metadata[:split_idx1]
    val_data = metadata[split_idx1:split_idx2]
    test_data = metadata[split_idx2:]
    
    print(f"📈 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
    
    # Create datasets
    train_dataset = BreakHisExtendedDataset(train_data, transform=transform, target_transform=target_transform)
    val_dataset = BreakHisExtendedDataset(val_data, transform=transform, target_transform=target_transform)
    test_dataset = BreakHisExtendedDataset(test_data, transform=transform, target_transform=target_transform)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"🔄 Batch size: {batch_size}, Train batches: {len(train_loader)}")
    
    # Models to train
    models_to_train = [
        'resnet50', 'resnet101', 'densenet169', 'densenet201', 'mobilenet_v3_large',
        'efficientnet_b1', 'efficientnet_b2', 'inception_v3', 'shufflenet_v2_x1_5', 'squeezenet1_1'
    ]
    
    # Training results storage
    training_results = {}
    evaluation_results = {}
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train each model
    for model_name in models_to_train:
        try:
            print(f"\n🎯 Training {model_name.upper()}...")
            
            # Train model
            model, results = train_extended_model(
                model_name, train_loader, val_loader, 
                num_epochs=8, learning_rate=0.001
            )
            
            training_results[model_name] = results
            
            # Evaluate on test set
            print(f"🧪 Evaluating {model_name} on test set...")
            test_metrics = evaluate_extended_model(model, test_loader, model_name)
            evaluation_results[model_name] = test_metrics
            
            print(f"✅ {model_name} completed!")
            print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Test F1-Score: {test_metrics['f1']:.4f}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    # Create performance summary
    print("\n🏆 FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    summary_data = []
    for model_name, metrics in evaluation_results.items():
        summary_data.append({
            'Model': model_name,
            'Test Accuracy': f"{metrics['accuracy']:.4f}",
            'Test Precision': f"{metrics['precision']:.4f}",
            'Test Recall': f"{metrics['recall']:.4f}",
            'Test F1-Score': f"{metrics['f1']:.4f}",
            'Best Val Acc': f"{training_results[model_name]['best_val_acc']:.2f}%"
        })
    
    # Display summary table
    for item in summary_data:
        print(f"{item['Model']:20} | Acc: {item['Test Accuracy']:6} | F1: {item['Test F1-Score']:6} | Val: {item['Best Val Acc']:6}")
    
    # Save results
    results_file = 'extended_training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'summary': summary_data
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎉 Extended model training completed successfully!")
    
    # List all available models
    print("\n📁 Available Models:")
    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
    for model_file in sorted(model_files):
        print(f"   • {model_file}")

if __name__ == "__main__":
    main()
