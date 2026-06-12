import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import os
import json
from tqdm import tqdm
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
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Split data
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(self.metadata)
        
        n_total = len(self.metadata)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        if split == 'train':
            self.images = self.metadata[:n_train]
        elif split == 'val':
            self.images = self.metadata[n_train:n_train + n_val]
        else:  # test
            self.images = self.metadata[n_train + n_val:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        image_path = os.path.join(self.data_dir, item['image_path'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get label (0 for benign, 1 for malignant)
        label = 0 if item['class'] == 'benign' else 1
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(architecture, num_classes=2):
    """Create different CNN models"""
    if architecture == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif architecture == 'densenet169':
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif architecture == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
    elif architecture == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

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
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
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
        
        # Clear GPU cache periodically and monitor performance
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            monitor_gpu_performance()
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
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

def extract_features(model, dataloader, device):
    """Extract features from CNN model for traditional ML"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, label in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            # Remove the last classification layer to get features
            if hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    features_batch = model.classifier[:-1](model.features(images))
                else:
                    features_batch = model.features(images)
            elif hasattr(model, 'fc'):
                # For ResNet models, get features before the final layer
                features_batch = model.avgpool(
                    model.layer4(
                        model.layer3(
                            model.layer2(
                                model.layer1(
                                    model.conv1(images)
                                )
                            )
                        )
                    )
                )
            else:
                features_batch = model.features(images)
            
            # Flatten features
            features_batch = features_batch.view(features_batch.size(0), -1)
            features.append(features_batch.cpu().numpy())
            labels.append(label.numpy())
    
    return np.vstack(features), np.concatenate(labels)

def train_traditional_ml_models(X_train, y_train, X_val, y_val):
    """Train traditional ML models"""
    models_dict = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted')
        val_recall = recall_score(y_val, y_val_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        
        print(f"{name}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    return results

def plot_training_curves(cnn_models, ml_models, save_dir='results'):
    """Plot training curves for all CNN models"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create multiple detailed plots for research paper
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Training Analysis for Breast Cancer Detection Models', fontsize=18, fontweight='bold')
    
    # 1. Training Loss Curves
    ax1 = axes[0, 0]
    for name, data in cnn_models.items():
        ax1.plot(data['train_losses'], label=f'{name}', linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss Curves
    ax2 = axes[0, 1]
    for name, data in cnn_models.items():
        ax2.plot(data['val_losses'], label=f'{name}', linewidth=2, alpha=0.8)
    ax2.set_title('Validation Loss Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Accuracy Curves
    ax3 = axes[0, 2]
    for name, data in cnn_models.items():
        ax3.plot(data['train_accs'], label=f'{name}', linewidth=2, alpha=0.8)
    ax3.set_title('Training Accuracy Evolution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Accuracy Curves
    ax4 = axes[1, 0]
    for name, data in cnn_models.items():
        ax4.plot(data['val_accs'], label=f'{name}', linewidth=2, alpha=0.8)
    ax4.set_title('Validation Accuracy Evolution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Final Validation Accuracy Comparison
    ax5 = axes[1, 1]
    names = list(cnn_models.keys())
    val_accs = [cnn_models[name]['best_val_acc'] for name in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax5.bar(names, val_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax5.set_title('Final Validation Accuracy by Model', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Model Architecture', fontsize=12)
    ax5.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax5.tick_params(axis='x', rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, val_accs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. All Models Performance Comparison (CNN + ML)
    ax6 = axes[1, 2]
    all_models = list(cnn_models.keys()) + list(ml_models.keys())
    all_accs = [cnn_models[name]['best_val_acc'] for name in cnn_models.keys()]
    
    for name, data in ml_models.items():
        all_accs.append(data['val_acc'] * 100)
    
    # Color coding: CNN models in blue, ML models in green
    cnn_colors = ['#1f77b4'] * len(cnn_models)
    ml_colors = ['#2ca02c'] * len(ml_models)
    all_colors = cnn_colors + ml_colors
    
    bars = ax6.bar(range(len(all_models)), all_accs, color=all_colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    ax6.set_title('All Models Performance Comparison', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Model', fontsize=12)
    ax6.set_ylabel('Accuracy (%)', fontsize=12)
    ax6.set_xticks(range(len(all_models)))
    ax6.set_xticklabels(all_models, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, all_accs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='CNN Models'),
                      Patch(facecolor='#2ca02c', label='Traditional ML Models')]
    ax6.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed plots
    # 1. Learning Rate Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in cnn_models.items():
        epochs = range(1, len(data['train_losses']) + 1)
        ax.plot(epochs, data['train_losses'], label=f'{name} (Train)', linewidth=2, alpha=0.8)
        ax.plot(epochs, data['val_losses'], label=f'{name} (Val)', linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_title('Learning Rate and Convergence Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Model Complexity vs Performance
    fig, ax = plt.subplots(figsize=(12, 8))
    model_params = {
        'VGG16': 138.4, 'VGG19': 143.7, 'ResNet18': 11.7, 'ResNet34': 21.8, 'ResNet50': 25.6,
        'DenseNet121': 8.1, 'DenseNet169': 14.3, 'MobileNetV2': 3.5, 'MobileNetV3_small': 2.5,
        'EfficientNetB0': 5.3, 'EfficientNetB3': 12.0, 'InceptionV3': 27.2,
        'ShuffleNetV2_x1_0': 2.3, 'SqueezeNet1_1': 1.2
    }
    
    x_pos = []
    y_pos = []
    labels = []
    
    for name in cnn_models.keys():
        if name.upper() in model_params:
            x_pos.append(model_params[name.upper()])
            y_pos.append(cnn_models[name]['best_val_acc'])
            labels.append(name)
    
    scatter = ax.scatter(x_pos, y_pos, c=range(len(x_pos)), cmap='viridis', s=100, alpha=0.7)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (x_pos[i], y_pos[i]), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold')
    
    ax.set_title('Model Complexity vs Performance Trade-off', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model Parameters (Millions)', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Model Index', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complexity_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(cnn_models, ml_models, test_loader, device, save_dir='results'):
    """Plot confusion matrices for all models with enhanced research quality"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Test CNN models
    cnn_results = {}
    for name, data in cnn_models.items():
        model = data['model']
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {name}"):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        cnn_results[name] = {
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    # Test ML models
    ml_results = {}
    if ml_models:
        # Extract features using first CNN
        first_cnn = list(cnn_models.values())[0]['model']
        features, labels = extract_features(first_cnn, test_loader, device)
        
        for name, data in ml_models.items():
            model = data['model']
            predictions = model.predict(features)
            ml_results[name] = {
                'predictions': predictions,
                'true_labels': labels
            }
    
    # Plot confusion matrices with enhanced styling
    all_results = {**cnn_results, **ml_results}
    n_models = len(all_results)
    cols = 4
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, results) in enumerate(all_results.items()):
        row = idx // cols
        col = idx % cols
        
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        
        # Calculate metrics for annotation
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Enhanced heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'],
                   ax=axes[row, col], cbar=True, square=True,
                   annot_kws={'size': 12, 'weight': 'bold'})
        
        axes[row, col].set_title(f'{name}\nConfusion Matrix', fontsize=14, fontweight='bold', pad=20)
        axes[row, col].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[row, col].set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add metrics text
        metrics_text = f'Acc: {accuracy:.3f}\nSens: {sensitivity:.3f}\nSpec: {specificity:.3f}'
        axes[row, col].text(0.02, 0.98, metrics_text, transform=axes[row, col].transAxes, 
                           fontsize=10, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Confusion Matrix Analysis for All Models', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional confusion matrix analysis
    # 1. Overall performance comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    model_names = []
    accuracies = []
    sensitivities = []
    specificities = []
    
    for name, results in all_results.items():
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        model_names.append(name)
        accuracies.append(accuracy * 100)
        sensitivities.append(sensitivity * 100)
        specificities.append(specificity * 100)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, sensitivities, width, label='Sensitivity', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, specificities, width, label='Specificity', color='#F18F01', alpha=0.8)
    
    ax.set_title('Comprehensive Model Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROC-like analysis (using confusion matrix data)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, results in all_results.items():
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        tn, fp, fn, tp = cm.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ax.scatter(fpr, tpr, s=100, alpha=0.7, label=name)
        ax.annotate(name, (fpr, tpr), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold')
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    ax.set_title('Model Performance in ROC Space', fontsize=16, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_space_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_table(cnn_models, ml_models, test_loader, device, save_dir='results'):
    """Generate comprehensive performance table"""
    os.makedirs(save_dir, exist_ok=True)
    
    results_data = []
    
    # Test CNN models
    for name, data in cnn_models.items():
        model = data['model']
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating {name}"):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        results_data.append({
            'Model': name,
            'Type': 'CNN',
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100,
            'Validation_Acc': data['best_val_acc']
        })
    
    # Test ML models
    if ml_models:
        # Extract features using first CNN
        first_cnn = list(cnn_models.values())[0]['model']
        features, labels = extract_features(first_cnn, test_loader, device)
        
        for name, data in ml_models.items():
            predictions = data['model'].predict(features)
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted')
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')
            
            results_data.append({
                'Model': name,
                'Type': 'Traditional ML',
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1 * 100,
                'Validation_Acc': data['val_acc'] * 100
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(results_data)
    df = df.sort_values('Accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(os.path.join(save_dir, 'performance_results.csv'), index=False)
    
    # Create formatted table
    fig, ax = plt.subplots(figsize=(12, len(results_data) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Comprehensive Model Performance Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'performance_table.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

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

def generate_additional_research_diagrams(cnn_models, ml_models, performance_df, save_dir='results'):
    """Generate additional research-quality diagrams for comprehensive analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔬 Generating additional research diagrams...")
    
    # 1. Model Architecture Comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Model parameters and performance
    model_info = {
        'VGG16': {'params': 138.4, 'type': 'CNN', 'depth': 'Deep'},
        'VGG19': {'params': 143.7, 'type': 'CNN', 'depth': 'Deep'},
        'ResNet18': {'params': 11.7, 'type': 'CNN', 'depth': 'Medium'},
        'ResNet34': {'params': 21.8, 'type': 'CNN', 'depth': 'Medium'},
        'ResNet50': {'params': 25.6, 'type': 'CNN', 'depth': 'Medium'},
        'DenseNet121': {'params': 8.1, 'type': 'CNN', 'depth': 'Medium'},
        'DenseNet169': {'params': 14.3, 'type': 'CNN', 'depth': 'Medium'},
        'MobileNetV2': {'params': 3.5, 'type': 'CNN', 'depth': 'Light'},
        'MobileNetV3_small': {'params': 2.5, 'type': 'CNN', 'depth': 'Light'},
        'EfficientNetB0': {'params': 5.3, 'type': 'CNN', 'depth': 'Light'},
        'EfficientNetB3': {'params': 12.0, 'type': 'CNN', 'depth': 'Medium'},
        'InceptionV3': {'params': 27.2, 'type': 'CNN', 'depth': 'Deep'},
        'ShuffleNetV2_x1_0': {'params': 2.3, 'type': 'CNN', 'depth': 'Light'},
        'SqueezeNet1_1': {'params': 1.2, 'type': 'CNN', 'depth': 'Light'}
    }
    
    # Get performance data
    x_pos = []
    y_pos = []
    colors = []
    sizes = []
    labels = []
    
    for name in cnn_models.keys():
        if name.upper() in model_info:
            info = model_info[name.upper()]
            perf = performance_df[performance_df['Model'] == name]['Accuracy'].iloc[0] if len(performance_df[performance_df['Model'] == name]) > 0 else 0
            
            x_pos.append(info['params'])
            y_pos.append(perf)
            
            # Color by depth
            if info['depth'] == 'Deep':
                colors.append('#FF6B6B')  # Red
            elif info['depth'] == 'Medium':
                colors.append('#4ECDC4')  # Teal
            else:
                colors.append('#45B7D1')  # Blue
            
            # Size by performance
            sizes.append(100 + perf * 2)
            labels.append(name)
    
    scatter = ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (x_pos[i], y_pos[i]), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold')
    
    ax.set_title('Model Architecture Analysis: Parameters vs Performance', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model Parameters (Millions)', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend for depth
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Deep Networks'),
                      Patch(facecolor='#4ECDC4', label='Medium Networks'),
                      Patch(facecolor='#45B7D1', label='Light Networks')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'architecture_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Training Convergence Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Convergence Analysis', fontsize=18, fontweight='bold')
    
    # Early convergence analysis
    ax1 = axes[0, 0]
    for name, data in cnn_models.items():
        epochs = range(1, len(data['train_losses']) + 1)
        ax1.plot(epochs, data['train_losses'], label=name, linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Convergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Validation convergence
    ax2 = axes[0, 1]
    for name, data in cnn_models.items():
        epochs = range(1, len(data['val_losses']) + 1)
        ax2.plot(epochs, data['val_losses'], label=name, linewidth=2, alpha=0.8)
    ax2.set_title('Validation Loss Convergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Overfitting analysis
    ax3 = axes[1, 0]
    for name, data in cnn_models.items():
        epochs = range(1, len(data['train_accs']) + 1)
        gap = [data['train_accs'][i] - data['val_accs'][i] for i in range(len(data['train_accs']))]
        ax3.plot(epochs, gap, label=name, linewidth=2, alpha=0.8)
    ax3.set_title('Overfitting Analysis (Train-Val Gap)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Gap (%)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Learning efficiency
    ax4 = axes[1, 1]
    for name, data in cnn_models.items():
        epochs = range(1, len(data['val_accs']) + 1)
        ax4.plot(epochs, data['val_accs'], label=name, linewidth=2, alpha=0.8)
    ax4.set_title('Learning Efficiency (Validation Accuracy)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Statistical Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Performance Analysis', fontsize=18, fontweight='bold')
    
    # Performance distribution
    ax1 = axes[0, 0]
    accuracies = performance_df['Accuracy'].values
    ax1.hist(accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.2f}%')
    ax1.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.2f}%')
    ax1.set_title('Performance Distribution')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Model type comparison
    ax2 = axes[0, 1]
    cnn_perf = performance_df[performance_df['Type'] == 'CNN']['Accuracy'].values
    ml_perf = performance_df[performance_df['Type'] == 'Traditional ML']['Accuracy'].values
    
    ax2.boxplot([cnn_perf, ml_perf], labels=['CNN Models', 'Traditional ML'], 
                patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_title('Performance by Model Type')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    # Correlation analysis
    ax3 = axes[1, 0]
    if len(cnn_models) > 1:
        # Create correlation matrix for CNN models
        cnn_data = []
        for name, data in cnn_models.items():
            cnn_data.append({
                'train_loss': np.mean(data['train_losses']),
                'val_loss': np.mean(data['val_losses']),
                'train_acc': np.mean(data['train_accs']),
                'val_acc': np.mean(data['val_accs'])
            })
        
        cnn_df = pd.DataFrame(cnn_data)
        correlation_matrix = cnn_df.corr()
        
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_title('Training Metrics Correlation')
        ax3.set_xticks(range(len(correlation_matrix.columns)))
        ax3.set_yticks(range(len(correlation_matrix.columns)))
        ax3.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax3.set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3)
    
    # Performance ranking
    ax4 = axes[1, 1]
    top_10 = performance_df.head(10)
    bars = ax4.barh(range(len(top_10)), top_10['Accuracy'], color='lightcoral', alpha=0.8)
    ax4.set_yticks(range(len(top_10)))
    ax4.set_yticklabels(top_10['Model'])
    ax4.set_title('Top 10 Performing Models')
    ax4.set_xlabel('Accuracy (%)')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Research Summary Dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Summary statistics
    ax1 = fig.add_subplot(gs[0, :2])
    summary_stats = [
        f"Total Models: {len(performance_df)}",
        f"CNN Models: {len(cnn_models)}",
        f"ML Models: {len(ml_models)}",
        f"Best Accuracy: {performance_df['Accuracy'].max():.2f}%",
        f"Average Accuracy: {performance_df['Accuracy'].mean():.2f}%",
        f"Dataset Size: {len(cnn_models) * 8 * 32 if cnn_models else 0} images"
    ]
    
    ax1.text(0.1, 0.9, 'Research Summary', fontsize=16, fontweight='bold', transform=ax1.transAxes)
    for i, stat in enumerate(summary_stats):
        ax1.text(0.1, 0.8 - i*0.12, stat, fontsize=12, transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Performance heatmap
    ax2 = fig.add_subplot(gs[0, 2:])
    performance_matrix = performance_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
    im = ax2.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(performance_matrix[0])))
    ax2.set_yticks(range(len(performance_matrix)))
    ax2.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], rotation=45)
    ax2.set_yticklabels(performance_df['Model'])
    plt.colorbar(im, ax=ax2)
    
    # Model type distribution
    ax3 = fig.add_subplot(gs[1, :2])
    type_counts = performance_df['Type'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax3.set_title('Model Type Distribution', fontsize=14, fontweight='bold')
    
    # Top performers
    ax4 = fig.add_subplot(gs[1, 2:])
    top_5 = performance_df.head(5)
    bars = ax4.bar(range(len(top_5)), top_5['Accuracy'], color='lightblue', alpha=0.8)
    ax4.set_title('Top 5 Performing Models', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xticks(range(len(top_5)))
    ax4.set_xticklabels([f'#{i+1}' for i in range(len(top_5))])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Training curves summary
    ax5 = fig.add_subplot(gs[2:, :])
    for name, data in cnn_models.items():
        epochs = range(1, len(data['val_accs']) + 1)
        ax5.plot(epochs, data['val_accs'], label=name, linewidth=2, alpha=0.8)
    ax5.set_title('Validation Accuracy Evolution for All Models', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Validation Accuracy (%)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Research Dashboard for Breast Cancer Detection Models', 
                fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'research_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ All additional research diagrams generated successfully!")

def main():
    """Main training function"""
    print("🚀 Starting Comprehensive Model Training for Breast Cancer Detection")
    print("=" * 70)
    
    # Set device and optimize CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # CUDA Performance Optimizations
        torch.backends.cudnn.benchmark = True  # Faster convolutions
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat-32 for faster matrix multiplications
        torch.backends.cudnn.allow_tf32 = True  # Allow TensorFloat-32 for convolutions
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
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
    
    # Create metadata if it doesn't exist
    if not os.path.exists(metadata_file):
        from train_unet import create_simple_metadata
        print("Creating metadata file...")
        create_simple_metadata(data_dir, metadata_file)
    
    train_dataset = BreakHisClassificationDataset(data_dir, metadata_file, transform, 'train')
    val_dataset = BreakHisClassificationDataset(data_dir, metadata_file, transform, 'val')
    test_dataset = BreakHisClassificationDataset(data_dir, metadata_file, transform, 'test')
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Total images: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"Training on entire dataset with optimized batch size")
    
    # Create dataloaders - optimized for CUDA performance
    # Calculate optimal batch size based on GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory >= 8:  # RTX 3070 Ti has 8GB+
        optimal_batch_size = 64
        num_workers = 4
    elif gpu_memory >= 6:
        optimal_batch_size = 48
        num_workers = 3
    else:
        optimal_batch_size = 32
        num_workers = 2
    
    print(f"🚀 GPU Memory: {gpu_memory:.1f}GB → Optimal Batch Size: {optimal_batch_size}, Workers: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=optimal_batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    # CNN architectures to train
    cnn_architectures = [
        'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
        'densenet121', 'densenet169', 'mobilenet_v2', 'mobilenet_v3_small',
        'efficientnet_b0', 'efficientnet_b3', 'inception_v3',
        'shufflenet_v2_x1_0', 'squeezenet1_1'
    ]
    
    # Train CNN models
    print("\n🔬 Training CNN Models...")
    print("=" * 50)
    
    cnn_models = {}
    for arch in cnn_architectures:
        try:
            print(f"\nTraining {arch.upper()}...")
            model = create_model(arch)
            results = train_cnn_model(model, train_loader, val_loader, device, num_epochs=8, model_name=arch.upper())
            cnn_models[arch.upper()] = results
            
            # Save model
            torch.save(results['model'].state_dict(), f'models/{arch}_breast_cancer.pth')
            print(f"✅ {arch.upper()} trained and saved!")
            
        except Exception as e:
            print(f"❌ Error training {arch}: {str(e)}")
            continue
    
    # Train traditional ML models
    print("\n🌳 Training Traditional ML Models...")
    print("=" * 50)
    
    # Extract features using the best CNN model
    best_cnn = max(cnn_models.items(), key=lambda x: x[1]['best_val_acc'])
    print(f"Using {best_cnn[0]} for feature extraction")
    
    # Extract features for ML training
    print("Extracting features for ML models...")
    X_train, y_train = extract_features(best_cnn[1]['model'], train_loader, device)
    X_val, y_val = extract_features(best_cnn[1]['model'], val_loader, device)
    
    # Train ML models
    ml_models = train_traditional_ml_models(X_train, y_train, X_val, y_val)
    
    # Generate comprehensive results
    print("\n📊 Generating Research-Quality Results...")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate all visualizations
    print("📊 Generating comprehensive research-quality visualizations...")
    plot_training_curves(cnn_models, ml_models, 'results')
    plot_confusion_matrices(cnn_models, ml_models, test_loader, device, 'results')
    performance_df = generate_performance_table(cnn_models, ml_models, test_loader, device, 'results')
    
    # Generate additional research diagrams
    generate_additional_research_diagrams(cnn_models, ml_models, performance_df, 'results')
    
    # Save all results
    print("\n💾 Saving Results...")
    print("=" * 30)
    
    # Save model states
    for name, data in cnn_models.items():
        torch.save(data['model'].state_dict(), f'models/{name.lower()}_breast_cancer.pth')
    
    # Save performance metrics
    performance_df.to_csv('results/comprehensive_results.csv', index=False)
    
    # Save training history
    training_history = {}
    for name, data in cnn_models.items():
        training_history[name] = {
            'train_losses': data['train_losses'],
            'val_losses': data['val_losses'],
            'train_accs': data['train_accs'],
            'val_accs': data['val_accs']
        }
    
    import pickle
    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    
    print("\n🎉 Training Complete! All models trained and results generated.")
    print("📁 Results saved in 'results/' directory")
    print("🤖 Models saved in 'models/' directory")
    print("📊 Performance table saved as 'results/comprehensive_results.csv'")
    
    # Print top 5 models
    print("\n🏆 Top 5 Performing Models:")
    print("=" * 40)
    top_5 = performance_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"{idx+1}. {row['Model']}: {row['Accuracy']:.2f}% (F1: {row['F1-Score']:.2f}%)")

if __name__ == "__main__":
    main()
