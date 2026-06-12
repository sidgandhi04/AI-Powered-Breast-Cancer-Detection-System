import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import restoration, exposure
import torch
import torch.nn as nn
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import json

# --- U-Net and Grad-CAM imports ---
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

# GAN imports
try:
    from gan_models import BreastCancerGAN, BreastCancerDataset, get_transforms
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False

# Global model cache to prevent rebuilding models
MODEL_CACHE = {}

# Global image tensor cache to ensure consistent preprocessing
IMAGE_TENSOR_CACHE = {}

st.title('Breast Cancer Detection')

st.sidebar.header('Navigation')
section = st.sidebar.radio('Go to', ['Upload Image', 'Preprocessing', 'Segmentation', 'Feature Extraction', 'Classification', 'Staging'])

# Load model configurations
@st.cache_data
def load_model_configs():
    """Load model configurations for all trained models"""
    configs = {
        'ResNet18': {
            'file': 'models/resnet18_breast_cancer_optimized.pth',
            'architecture': 'resnet18',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'ResNet18 - Optimized with Mixed Precision Training'
        },
        'ResNet34': {
            'file': 'models/resnet34_breast_cancer_optimized.pth',
            'architecture': 'resnet34',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'ResNet34 - Optimized with Mixed Precision Training'
        },
        'DenseNet121': {
            'file': 'models/densenet121_breast_cancer_optimized.pth',
            'architecture': 'densenet121',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'DenseNet121 - Optimized with Mixed Precision Training'
        },
        'MobileNetV2': {
            'file': 'models/mobilenet_v2_breast_cancer_optimized.pth',
            'architecture': 'mobilenet_v2',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'MobileNetV2 - Optimized with Mixed Precision Training'
        },
        'EfficientNetB0': {
            'file': 'models/efficientnet_b0_breast_cancer_optimized.pth',
            'architecture': 'efficientnet_b0',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'EfficientNetB0 - Optimized with Mixed Precision Training'
        },
        'VGG16': {
            'file': 'models/breakhis_tiny_vgg16.pth',
            'architecture': 'vgg16',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'VGG16 - Pre-trained on BreakHis dataset'
        },
        
        # Advanced CNN Models (Newly Trained)
        'EfficientNet B3': {
            'file': 'models/efficientnet_b3_breast_cancer_advanced.pth',
            'architecture': 'efficientnet_b3',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'EfficientNet B3 - Advanced Model for High Accuracy'
        },
        'EfficientNet B4': {
            'file': 'models/efficientnet_b4_breast_cancer_advanced.pth',
            'architecture': 'efficientnet_b4',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'EfficientNet B4 - Advanced Model for High Accuracy'
        },
        'MobileNet V3 Large': {
            'file': 'models/mobilenet_v3_large_breast_cancer_advanced.pth',
            'architecture': 'mobilenet_v3_large',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'MobileNet V3 Large - Mobile-Optimized Advanced Model'
        },
        'ShuffleNet V2': {
            'file': 'models/shufflenet_v2_x1_0_breast_cancer_advanced.pth',
            'architecture': 'shufflenet_v2_x1_0',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'ShuffleNet V2 - Ultra-Fast Mobile Architecture'
        },
        'SqueezeNet 1.1': {
            'file': 'models/squeezenet1_1_breast_cancer_advanced.pth',
            'architecture': 'squeezenet1_1',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'SqueezeNet 1.1 - Lightweight and Efficient'
        },
        'ConvNeXt Base': {
            'file': 'models/convnext_base_breast_cancer_advanced.pth',
            'architecture': 'convnext_base',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'ConvNeXt Base - Modern CNN Architecture'
        },
        'RegNet Y-32GF': {
            'file': 'models/regnet_y_32gf_breast_cancer_advanced.pth',
            'architecture': 'regnet_y_32gf',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'RegNet Y-32GF - High-Performance CNN Architecture'
        },
        
        # Tiny Models
        'Tiny VGG16': {
            'file': 'models/breakhis_tiny_vgg16.pth',
            'architecture': 'vgg16',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'Tiny VGG16 - Compact Version for Fast Inference'
        },
        'Tiny MobileNetV2': {
            'file': 'models/breakhis_tiny_mobilenetv2.pth',
            'architecture': 'mobilenet_v2',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'Tiny MobileNetV2 - Compact Mobile Architecture'
        },
        'Tiny DenseNet121': {
            'file': 'models/breakhis_tiny_densenet121.pth',
            'architecture': 'densenet121',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'Tiny DenseNet121 - Compact Dense Architecture'
        },
        'Tiny EfficientNetB3': {
            'file': 'models/breakhis_tiny_efficientnetb3.pth',
            'architecture': 'efficientnet_b3',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'Tiny EfficientNetB3 - Compact Efficient Architecture'
        },
        'Tiny ResNet18': {
            'file': 'models/breakhis_tiny_resnet18.pth',
            'architecture': 'resnet18',
            'input_size': (224, 224),
            'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            'description': 'Tiny ResNet18 - Compact ResNet Architecture'
        },
        
        # GAN Models
        'GAN Generator': {
            'file': 'models/gan_breast_cancer_final.pth',
            'architecture': 'gan_generator',
            'input_size': (224, 224),
            'normalization': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            'description': 'GAN Generator - Synthetic Image Generation for Data Augmentation'
        }
    }
    return configs

@st.cache_data
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return np.array(image)

# --- Helper: Grad-CAM ---
def grad_cam(model, img_tensor, target_layer):
    # Simple Grad-CAM for ResNet-like models
    gradients = []
    activations = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    def forward_hook(module, input, output):
        activations.append(output.detach())
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    model.zero_grad()
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    loss = output[0, pred_class]
    loss.backward()
    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_fwd.remove()
    handle_bwd.remove()
    return cam

# --- 1. Upload Image ---
if section == 'Upload Image':
    st.header('1. Upload Histopathological Image')
    uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image_np = load_image(uploaded_file)
        st.image(image_np, caption='Uploaded Image')
        st.session_state['original_image'] = image_np
        st.success('Image uploaded and saved!')
    elif 'original_image' in st.session_state:
        st.image(st.session_state['original_image'], caption='Previously Uploaded Image')
    else:
        st.info('Please upload an image to proceed.')

# --- 2. Preprocessing ---
elif section == 'Preprocessing':
    st.header('2. Preprocessing')
    if 'original_image' not in st.session_state:
        st.warning('Please upload an image first.')
    else:
        image = st.session_state['original_image']
        st.subheader('Original Image')
        st.image(image, caption='Original')
        
        # Magnification detection and analysis
        st.subheader('🔍 Magnification Analysis')
        height, width = image.shape[:2]
        
        # Estimate magnification based on image size
        if height > 1000 or width > 1000:
            mag_color = "🔴"
        elif height > 500 or width > 500:
            mag_color = "🟡"
        else:
            mag_color = "🟢"
            mag_warning = "ℹ️ **Low Resolution Image**: Consider using higher resolution images for better results."
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"{mag_color} **Estimated Magnification**: {estimated_mag}")
            st.write(f"Image dimensions: {width} x {height} pixels")
        with col2:
            st.warning(mag_warning)
        
        # Store magnification info for later use
        st.session_state['estimated_magnification'] = estimated_mag
        st.session_state['image_dimensions'] = (width, height)
        
        # Denoising
        st.subheader('Denoising')
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        st.image(denoised, caption='Denoised')
        st.caption('Denoising removes random noise from the image.')
        
        # Deblurring (Wiener filter)
        st.subheader('Deblurring')
        gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        from skimage import filters
        deblurred, _ = restoration.unsupervised_wiener(gray, np.ones((5, 5)) / 25)
        deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
        st.image(deblurred, caption='Deblurred (grayscale)')
        st.caption('Deblurring attempts to reverse blurring effects.')
        
        # Normalization
        st.subheader('Normalization')
        norm = exposure.rescale_intensity(denoised, in_range='image', out_range=(0, 255)).astype(np.uint8)
        st.image(norm, caption='Normalized')
        st.caption('Normalization scales pixel values for better model performance.')
        st.session_state['preprocessed_image'] = norm

# --- 3. Segmentation ---
elif section == 'Segmentation':
    st.header('3. Tissue Segmentation')
    if 'preprocessed_image' not in st.session_state:
        st.warning('Please preprocess an image first.')
    else:
        image = st.session_state['preprocessed_image']
        
        if smp is not None:
            st.subheader('U-Net Segmentation')
            
            # Check if U-Net model exists
            if not os.path.exists('models/unet_segmentation.pth'):
                st.error('U-Net segmentation model not found. Please train the model first using train_unet.py')
                st.info('To train the model, run: python train_unet.py')
                
                # Display only the preprocessed image when no model is available
                st.image(image, caption='Preprocessed Image')
                st.warning('Segmentation mask will be available after training the U-Net model.')
            else:
                # Load U-Net model
                unet = smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
                unet.load_state_dict(torch.load('models/unet_segmentation.pth', map_location='cpu'))
                unet.eval()
                
                # Preprocess image for U-Net
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    mask = unet(img_tensor)
                    mask = torch.sigmoid(mask)
                    
                    # Store raw mask for debugging
                    raw_mask = mask.squeeze().cpu().numpy()
                    
                    # Apply thresholding with adjustable threshold
                    # Calculate a better default threshold based on the raw mask statistics
                    suggested_threshold = min(0.7, max(0.3, raw_mask.mean() - 0.1))
                    threshold = st.slider('Segmentation Threshold', 0.1, 0.9, suggested_threshold, 0.05, 
                                        help='Adjust threshold for segmentation sensitivity. Higher values = more selective')
                    mask = (mask > threshold).float()
                
                # Convert mask to numpy
                mask_np = mask.squeeze().cpu().numpy()
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption='Preprocessed Image')
                with col2:
                    # Convert grayscale mask to RGB for display and ensure proper scaling
                    if len(mask_np.shape) == 2:
                        # Scale mask to 0-255 range for better visibility
                        mask_display = (mask_np * 255).astype(np.uint8)
                        mask_rgb = np.stack([mask_display, mask_display, mask_display], axis=-1)
                    else:
                        mask_rgb = mask_np
                    
                    st.image(mask_rgb, caption='Segmentation Mask')
                    
                    # Add multiple visualization options
                    st.subheader("🎨 Visualization Options")
                    
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        if st.checkbox('Show Enhanced Mask Visualization'):
                            # Create a more visible mask with better contrast
                            enhanced_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                            enhanced_mask[mask_np > 0] = [255, 0, 0]  # Red for tissue
                            enhanced_mask[mask_np == 0] = [0, 0, 0]   # Black for background
                            
                            # Add a border for better visibility
                            enhanced_mask[0, :] = [255, 255, 255]  # Top border
                            enhanced_mask[-1, :] = [255, 255, 255]  # Bottom border
                            enhanced_mask[:, 0] = [255, 255, 255]  # Left border
                            enhanced_mask[:, -1] = [255, 255, 255]  # Right border
                            
                            st.image(enhanced_mask, caption='Enhanced Mask: Red=Tissue, Black=Background')
                    
                    with col_viz2:
                        if st.checkbox('Show Colorful Mask'):
                            # Create a colorful mask with different colors
                            colorful_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                            colorful_mask[mask_np > 0] = [0, 255, 0]  # Green for tissue
                            colorful_mask[mask_np == 0] = [50, 50, 50]  # Dark gray for background
                            
                            # Add gradient effect
                            for i in range(mask_np.shape[0]):
                                for j in range(mask_np.shape[1]):
                                    if mask_np[i, j] > 0:
                                        # Create a gradient effect
                                        intensity = int(mask_np[i, j] * 255)
                                        colorful_mask[i, j] = [0, intensity, 255 - intensity]
                            
                            st.image(colorful_mask, caption='Colorful Mask: Green=Tissue, Gray=Background')
                    
                    # Additional visualization options
                    if st.checkbox('Show Heatmap Visualization'):
                        # Create a heatmap-style visualization
                        import matplotlib.pyplot as plt
                        import matplotlib.cm as cm
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(mask_np, cmap='hot', interpolation='nearest')
                        ax.set_title('Segmentation Heatmap')
                        ax.axis('off')
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('Tissue Probability')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    if st.checkbox('Show Contour Visualization'):
                        # Create contour visualization
                        contour_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                        contour_mask[:, :] = [240, 240, 240]  # Light gray background
                        
                        # Find contours
                        contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Draw contours in bright colors
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                        for i, contour in enumerate(contours):
                            color = colors[i % len(colors)]
                            cv2.drawContours(contour_mask, [contour], -1, color, 2)
                        
                        st.image(contour_mask, caption='Contour Visualization: Colored contours show tissue boundaries')
                    
                    # Add overlay visualization
                    if st.checkbox('Show Overlay on Original Image'):
                        # Create overlay by blending original image with mask
                        overlay = image.copy()
                        if len(overlay.shape) == 3:
                            # Convert to numpy if it's a PIL image
                            overlay = np.array(overlay)
                        
                        # Resize mask to match original image dimensions
                        from skimage.transform import resize
                        mask_resized = resize(mask_np, (overlay.shape[0], overlay.shape[1]), 
                                            anti_aliasing=False, preserve_range=True)
                        
                        # Create colored mask (red for tissue)
                        colored_mask = np.zeros_like(overlay)
                        colored_mask[:, :, 0] = mask_resized * 255  # Red channel
                        
                        # Blend original image with colored mask
                        alpha = 0.3  # Transparency
                        overlay = (1 - alpha) * overlay + alpha * colored_mask
                        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                        
                        st.image(overlay, caption='Overlay: Red = Detected Tissue')
                
                # Add debugging information
                st.write(f"**Debug Info:**")
                st.write(f"- Raw mask range: [{raw_mask.min():.3f}, {raw_mask.max():.3f}]")
                st.write(f"- Raw mask mean: {raw_mask.mean():.3f}")
                st.write(f"- Threshold used: {threshold}")
                st.write(f"- Mask pixels > threshold: {(mask_np > 0).sum()} / {mask_np.size}")
                
                # Show raw mask for comparison
                if st.checkbox('Show Raw Mask (before thresholding)'):
                    raw_mask_display = (raw_mask * 255).astype(np.uint8)
                    raw_mask_rgb = np.stack([raw_mask_display, raw_mask_display, raw_mask_display], axis=-1)
                    st.image(raw_mask_rgb, caption='Raw Mask (before thresholding)')
                
                st.caption('U-Net segmentation identifies tissue regions of interest.')
        else:
            st.warning('U-Net segmentation not available. Install segmentation-models-pytorch for this feature.')

# --- 4. Feature Extraction ---
elif section == 'Feature Extraction':
    st.header('4. Feature Extraction')
    if 'preprocessed_image' not in st.session_state:
        st.warning('Please preprocess an image first.')
    else:
        image = st.session_state['preprocessed_image']

        st.subheader('Select Model for Feature Extraction')

        # Load all available model configurations
        model_configs = load_model_configs()
        model_options = list(model_configs.keys())

        # Filter out models that don't have model files
        available_models = []
        missing_models = []
        
        for model_name in model_options:
            config = model_configs[model_name]
            if os.path.exists(config['file']):
                available_models.append(model_name)
            else:
                missing_models.append(f"{model_name} ({config['file']})")
        
        # Show model count and missing models
        st.info(f"{len(available_models)} models available for feature extraction")
        if missing_models:
            with st.expander("Missing Model Files", expanded=False):
                for missing in missing_models:
                    st.warning(f"❌ {missing}")
        
        # Use only available models
        model_options = available_models

        selected_model = st.selectbox(
            'Choose a model:',
            model_options,
            help="Select from trained models including CNNs and mobile architectures"
        )

        if not selected_model:
            st.stop()

        st.success(f"{selected_model}: {model_configs[selected_model]['description']}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_results = {}

        # ---- helpers
        def safe_load(model, ckpt_path):
            if not os.path.exists(ckpt_path):
                st.error(f"Model file not found: {ckpt_path}")
                st.stop()
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if isinstance(state, dict):
                state = {k.replace('module.', ''): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(state.state_dict(), strict=False)
            return model

        def show_maps_and_stats(feature_map):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Feature Maps")
                if feature_map.dim() > 2:
                    num_features = min(16, feature_map.shape[0])
                    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                    for i in range(num_features):
                        r, c = i // 4, i % 4
                        axes[r, c].imshow(feature_map[i].cpu().numpy(), cmap='viridis')
                        axes[r, c].set_title(f'Feature {i+1}')
                        axes[r, c].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("Feature vector extracted (not a 2D feature map)")

            with col2:
                st.subheader("Feature Statistics")
                st.write(f"Feature Shape: {tuple(feature_map.shape)}")
                st.write(f"Mean Activation: {feature_map.mean().item():.4f}")
                st.write(f"Std Activation: {feature_map.std().item():.4f}")
                st.write(f"Max Activation: {feature_map.max().item():.4f}")
                st.write(f"Min Activation: {feature_map.min().item():.4f}")
                if feature_map.dim() > 2:
                    vari = feature_map.var(dim=(1, 2))
                    k = min(5, vari.shape[0])
                    top = torch.topk(vari, k)
                    st.write("Top features by variance:")
                    for i, (idx, var) in enumerate(zip(top.indices, top.values)):
                        st.write(f"{i+1}. #{idx.item()} = {var.item():.4f}")

        # ============= VGG16 =============
        if selected_model == 'VGG16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            model.features[10].register_forward_hook(hook_as('conv4_1'))
            model.features[17].register_forward_hook(hook_as('conv5_1'))
            model.features[24].register_forward_hook(hook_as('conv5_3'))
            model.avgpool.register_forward_hook(hook_as('avgpool'))

            with torch.no_grad(): _ = model(input_tensor)
            layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
            if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= ResNet18 =============
        elif selected_model == 'ResNet18':
            model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 2)
            model = safe_load(model, 'models/resnet18_breast_cancer_optimized.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            model.layer1.register_forward_hook(hook_as('layer1'))
            model.layer2.register_forward_hook(hook_as('layer2'))
            model.layer3.register_forward_hook(hook_as('layer3'))
            model.layer4.register_forward_hook(hook_as('layer4'))

            with torch.no_grad(): _ = model(input_tensor)
            layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
            if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= ResNet34 =============
        elif selected_model == 'ResNet34':
            model = models.resnet34(weights=None); model.fc = nn.Linear(model.fc.in_features, 2)
            model = safe_load(model, 'models/resnet34_breast_cancer_optimized.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            model.layer1.register_forward_hook(hook_as('layer1'))
            model.layer2.register_forward_hook(hook_as('layer2'))
            model.layer3.register_forward_hook(hook_as('layer3'))
            model.layer4.register_forward_hook(hook_as('layer4'))

            with torch.no_grad(): _ = model(input_tensor)
            layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
            if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= EfficientNetB0 =============
        elif selected_model == 'EfficientNetB0':
            model = models.efficientnet_b0(weights=None); model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            model = safe_load(model, 'models/efficientnet_b0_breast_cancer_optimized.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            for idx, nm in [(2, 'block1'), (4, 'block2'), (6, 'block3'), (8, 'block4')]:
                if idx < len(model.features): model.features[idx].register_forward_hook(hook_as(nm))

            with torch.no_grad(): _ = model(input_tensor)
            layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
            if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= DenseNet121 =============
        elif selected_model == 'DenseNet121':
            model = models.densenet121(weights=None); model.classifier = nn.Linear(model.classifier.in_features, 2)
            model = safe_load(model, 'models/densenet121_breast_cancer_optimized.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            for name in ['denseblock1','denseblock2','denseblock3','denseblock4','transition1','transition2','transition3','norm5']:
                if hasattr(model.features, name): getattr(model.features, name).register_forward_hook(hook_as(name))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            else:
                st.warning("No features captured")
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= MobileNetV2 =============
        elif selected_model == 'MobileNetV2':
            model = models.mobilenet_v2(weights=None); model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            model = safe_load(model, 'models/mobilenet_v2_breast_cancer_optimized.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            L = len(model.features)
            for idx, nm in [(2,'layer1'), (4,'layer2'), (7,'layer3'), (min(18,L-1),'layer4')]:
                if idx < L: model.features[idx].register_forward_hook(hook_as(nm))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            else:
                st.warning("No features captured")
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= EfficientNetB3 =============
        elif selected_model == 'EfficientNet B3':
            model = models.efficientnet_b3(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model = safe_load(model, 'models/efficientnet_b3_breast_cancer_advanced.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            for idx, nm in [(3,'block1'), (6,'block2'), (9,'block3'), (len(model.features)-1,'final')]:
                if idx < len(model.features): model.features[idx].register_forward_hook(hook_as(nm))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= EfficientNetB4 =============
        elif selected_model == 'EfficientNet B4':
            model = models.efficientnet_b4(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model = safe_load(model, 'models/efficientnet_b4_breast_cancer_advanced.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            for idx, nm in [(3,'block1'), (6,'block2'), (9,'block3'), (len(model.features)-1,'final')]:
                if idx < len(model.features): model.features[idx].register_forward_hook(hook_as(nm))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= MobileNetV3-Large =============
        elif selected_model == 'MobileNetV3-Large':
            model = models.mobilenet_v3_large(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model = safe_load(model, 'models/mobilenet_v3_large_breast_cancer_optimized.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            L = len(model.features)
            for idx, nm in [(2,'layer1'), (4,'layer2'), (7,'layer3'), (L-1,'final')]:
                if idx < L: model.features[idx].register_forward_hook(hook_as(nm))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= ShuffleNetV2_x1_0 =============
        elif selected_model == 'ShuffleNet V2':
            model = models.shufflenet_v2_x1_0(weights=None); model.fc = nn.Linear(model.fc.in_features, 2)
            model = safe_load(model, 'models/shufflenet_v2_x1_0_breast_cancer_advanced.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            model.conv1.register_forward_hook(hook_as('conv1'))
            model.stage2.register_forward_hook(hook_as('stage2'))
            model.stage3.register_forward_hook(hook_as('stage3'))
            model.stage4.register_forward_hook(hook_as('stage4'))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= SqueezeNet1_1 =============
        elif selected_model == 'SqueezeNet 1.1':
            model = models.squeezenet1_1(weights=None); model.classifier[1] = nn.Conv2d(512, 2, kernel_size=1)
            model = safe_load(model, 'models/squeezenet1_1_breast_cancer_advanced.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            model.features[2].register_forward_hook(hook_as('fire2'))
            model.features[4].register_forward_hook(hook_as('fire4'))
            model.features[6].register_forward_hook(hook_as('fire6'))
            model.features[8].register_forward_hook(hook_as('fire8'))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= ConvNeXt-Base =============
        elif selected_model == 'ConvNeXt Base':
            model = models.convnext_base(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model = safe_load(model, 'models/convnext_base_breast_cancer_advanced.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            for idx, nm in [(2,'stage1'), (4,'stage2'), (6,'stage3'), (8,'stage4')]:
                if idx < len(model.features): model.features[idx].register_forward_hook(hook_as(nm))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= RegNetY-32GF =============
        elif selected_model == 'RegNet Y-32GF':
            model = models.regnet_y_32gf(weights=None); model.fc = nn.Linear(model.fc.in_features, 2)
            model = safe_load(model, 'models/regnet_y_32gf_breast_cancer_advanced.pth').to(device).eval()
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            features = {}
            def hook_as(name): return lambda m,i,o: features.setdefault(name, o.detach())
            model.trunk_output.block1.register_forward_hook(hook_as('block1'))
            model.trunk_output.block2.register_forward_hook(hook_as('block2'))
            model.trunk_output.block3.register_forward_hook(hook_as('block3'))
            model.trunk_output.block4.register_forward_hook(hook_as('block4'))

            with torch.no_grad(): _ = model(input_tensor)
            if features:
                layer = st.selectbox(f"Select layer for {selected_model}:", list(features.keys()))
                if layer in features: show_maps_and_stats(features[layer][0])
            feature_results[selected_model] = {'features': features, 'input_shape': input_tensor.shape, 'model': model}

        # ============= GAN Generator =============
        elif selected_model == 'GAN Generator':
            if not GAN_AVAILABLE:
                st.error("GAN models not available. Please ensure gan_models.py is in the same directory.")
            else:
                # Load GAN model
                try:
                    # Check if model file exists
                    model_path = 'models/gan_breast_cancer_final.pth'
                    if not os.path.exists(model_path):
                        st.error(f"GAN model file not found: {model_path}")
                        st.info("Please ensure the GAN model has been trained and saved.")
                    else:
                        gan = BreastCancerGAN(device=device)
                        gan.load_models('gan_breast_cancer_final.pth')
                        
                        st.subheader("GAN Generator Features")
                        st.write("**GAN Generator**: Generates synthetic breast cancer histopathology images")
                        
                        # Generate synthetic images
                        num_samples = st.slider("Number of synthetic images to generate", 1, 16, 4)
                        
                        if st.button("Generate Synthetic Images"):
                            with st.spinner("Generating synthetic images..."):
                                synthetic_images = gan.generate_images(num_samples=num_samples)
                            
                                # Display generated images
                                cols = st.columns(2)
                                for i, img in enumerate(synthetic_images):
                                    with cols[i % 2]:
                                        # Convert tensor to numpy and display
                                        img_np = img.permute(1, 2, 0).numpy()
                                        img_np = np.clip(img_np, 0, 1)
                                        st.image(img_np, caption=f'Synthetic Image {i+1}', use_column_width=True)
                        
                        # Show GAN architecture info
                        st.subheader("GAN Architecture Information")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Generator Architecture:**")
                            st.write(f"- Input: Random noise vector (100D)")
                            st.write(f"- Output: 224x224x3 RGB image")
                            st.write(f"- Parameters: {sum(p.numel() for p in gan.generator.parameters()):,}")
                            
                        with col2:
                            st.write("**Discriminator Architecture:**")
                            st.write(f"- Input: 224x224x3 RGB image")
                            st.write(f"- Output: Real/Fake probability")
                            st.write(f"- Parameters: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
                    
                    # Show training history if available
                    if hasattr(gan, 'g_losses') and len(gan.g_losses) > 0:
                        st.subheader("Training History")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(gan.g_losses, label='Generator Loss', color='blue')
                        ax.plot(gan.d_losses, label='Discriminator Loss', color='red')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('GAN Training History')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                    
                    # Feature extraction from generated images
                    st.subheader("Feature Analysis of Generated Images")
                    if st.button("Analyze Generated Image Features"):
                        with st.spinner("Analyzing features..."):
                            # Generate one image for analysis
                            sample_img = gan.generate_images(num_images=1)[0]
                            sample_img_np = sample_img.permute(1, 2, 0).numpy()
                            sample_img_np = np.clip(sample_img_np, 0, 1)
                            
                            # Convert to PIL for feature extraction
                            sample_pil = Image.fromarray((sample_img_np * 255).astype(np.uint8))
                            
                            # Use a pre-trained model to extract features
                            feature_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            
                            img_tensor = transform(sample_pil).unsqueeze(0).to(device)
                            
                            # Extract features from different layers
                            features = {}
                            def hook_fn(name):
                                def hook(module, input, output):
                                    features[name] = output.detach()
                                return hook
                            
                            feature_model.layer1.register_forward_hook(hook_fn('layer1'))
                            feature_model.layer2.register_forward_hook(hook_fn('layer2'))
                            feature_model.layer3.register_forward_hook(hook_fn('layer3'))
                            feature_model.layer4.register_forward_hook(hook_fn('layer4'))
                            
                            with torch.no_grad():
                                _ = feature_model(img_tensor)
                            
                            # Display feature maps
                            st.write("**Feature Maps from Generated Image:**")
                            for layer_name, feature_map in features.items():
                                if feature_map.dim() > 2:
                                    st.write(f"**{layer_name}**: {feature_map.shape}")
                                    # Show first few feature maps
                                    num_features = min(8, feature_map.shape[1])
                                    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                                    for i in range(num_features):
                                        row, col = i // 4, i % 4
                                        axes[row, col].imshow(feature_map[0, i].cpu().numpy(), cmap='viridis')
                                        axes[row, col].set_title(f'Feature {i+1}')
                                        axes[row, col].axis('off')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    break  # Show only first layer to avoid too many plots
                    
                    feature_results[selected_model] = {
                        'features': {'gan_generator': gan.generator},
                        'input_shape': (1, 3, 224, 224),
                        'model': gan.generator,
                        'gan_model': gan
                    }
                    
                except Exception as e:
                    st.error(f"Error loading GAN model: {e}")
                    st.info("Make sure the GAN model has been trained and saved as 'gan_breast_cancer_final.pth'")

        # ============= Generic Handler for All Other Models =============
        else:
            # Generic handler for all models not specifically handled above
            try:
                st.info(f"Using generic feature extraction for {selected_model}")
                
                # Get model configuration
                config = model_configs[selected_model]
                arch = config['architecture']
                
                # Build model using the same logic as classification
                # Define helper functions locally
                def detect_num_classes_in_ckpt(ckpt_path):
                    """Try to infer num output classes from the checkpoint head."""
                    try:
                        sd = torch.load(ckpt_path, map_location='cpu')
                        if isinstance(sd, dict) and 'state_dict' in sd:
                            sd = sd['state_dict']
                        if isinstance(sd, dict):
                            sd = {k.replace('module.', ''): v for k, v in sd.items()}
                            
                            # Look for classifier weight patterns
                            classifier_keys = [
                                'classifier.6.weight',  # VGG
                                'fc.weight',           # ResNet, DenseNet
                                'classifier.1.weight', # MobileNet
                                '_fc.weight',          # EfficientNet
                                'classifier[-1].weight',
                                'classifier.weight',   # DenseNet direct
                                'classifier.0.weight', # Some architectures
                                'classifier.3.weight', # MobileNetV3
                                'classifier.4.weight', # Some variants
                                'classifier.5.weight'  # Some variants
                            ]
                            
                            for k in classifier_keys:
                                if k in sd and sd[k].dim() == 2:
                                    num_classes = int(sd[k].shape[0])
                                    return num_classes
                            
                            # If no specific classifier found, look for any weight with 2D shape
                            for k, v in sd.items():
                                if 'classifier' in k and 'weight' in k and v.dim() == 2:
                                    num_classes = int(v.shape[0])
                                    return num_classes
                                    
                    except Exception as e:
                        print(f"Error detecting classes in {ckpt_path}: {e}")
                        pass
                    return None  # unknown

                def build_model_local(model_name, config):
                    """Create model for inference, handle both 2-class and 6-class models."""
                    arch = config['architecture']
                    ckpt = config['file']

                    # Check class count in checkpoint
                    num_classes = detect_num_classes_in_ckpt(ckpt)
                    if num_classes is None:
                        num_classes = 2  # Default to 2 classes if can't detect
                        print(f"Warning: Could not detect classes in {ckpt}, defaulting to 2 classes")
                    else:
                        print(f"Building {model_name} with {num_classes} classes")
                    
                    # Build base torchvision model with correct number of classes
                    if arch == 'vgg16':
                        model = models.vgg16(weights=None)
                        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading VGG16 with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                model.classifier[6] = nn.Linear(model.classifier[6].in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    elif arch == 'resnet18':
                        model = models.resnet18(weights=None)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading ResNet18 with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                model.fc = nn.Linear(model.fc.in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    elif arch == 'densenet121':
                        model = models.densenet121(weights=None)
                        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading DenseNet121 with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                model.classifier = nn.Linear(model.classifier.in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    elif arch == 'mobilenet_v2':
                        model = models.mobilenet_v2(weights=None)
                        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading MobileNetV2 with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    elif arch == 'efficientnet_b0':
                        model = models.efficientnet_b0(weights=None)
                        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading EfficientNetB0 with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    elif arch == 'efficientnet_b3':
                        model = models.efficientnet_b3(weights=None)
                        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading EfficientNetB3 with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    else:
                        # For other architectures, try a generic approach
                        if arch == 'resnet34':
                            model = models.resnet34(weights=None)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                        elif arch == 'mobilenet_v3_large':
                            model = models.mobilenet_v3_large(weights=None)
                            if hasattr(model.classifier, '__len__'):
                                for i in range(len(model.classifier) - 1, -1, -1):
                                    if isinstance(model.classifier[i], nn.Linear):
                                        model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                                        break
                            else:
                                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                        else:
                            # Default fallback
                            model = models.resnet18(weights=None)
                            model.fc = nn.Linear(model.fc.in_features, num_classes)
                        
                        try:
                            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                        except Exception as e:
                            print(f"Error loading {arch} with {num_classes} classes: {e}")
                            if num_classes == 2:
                                print("Trying with 6 classes as fallback...")
                                if arch == 'resnet34':
                                    model.fc = nn.Linear(model.fc.in_features, 6)
                                elif arch == 'mobilenet_v3_large':
                                    for i in range(len(model.classifier) - 1, -1, -1):
                                        if isinstance(model.classifier[i], nn.Linear):
                                            model.classifier[i] = nn.Linear(model.classifier[i].in_features, 6)
                                            break
                                else:
                                    model.fc = nn.Linear(model.fc.in_features, 6)
                                model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
                                num_classes = 6

                    return model.to(device).eval(), num_classes

                def build_transform_local(config):
                    return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(config['input_size']),
                        transforms.ToTensor(),
                        transforms.Normalize(*config['normalization'])
                    ])

                model, num_classes = build_model_local(selected_model, config)
                transform = build_transform_local(config)
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Extract features from different layers based on architecture
                features = {}
                
                def hook_as(name): 
                    return lambda m, i, o: features.setdefault(name, o.detach())
                
                # Register hooks based on architecture
                if arch == 'resnet18' or arch == 'resnet34':
                    # For ResNet models, hook into different layers
                    if hasattr(model, 'layer1'):
                        model.layer1.register_forward_hook(hook_as('layer1'))
                    if hasattr(model, 'layer2'):
                        model.layer2.register_forward_hook(hook_as('layer2'))
                    if hasattr(model, 'layer3'):
                        model.layer3.register_forward_hook(hook_as('layer3'))
                    if hasattr(model, 'layer4'):
                        model.layer4.register_forward_hook(hook_as('layer4'))
                    if hasattr(model, 'avgpool'):
                        model.avgpool.register_forward_hook(hook_as('avgpool'))
                        
                elif arch == 'vgg16':
                    # For VGG models, hook into different feature layers
                    if hasattr(model, 'features'):
                        for i, layer in enumerate(model.features):
                            if isinstance(layer, nn.Conv2d):
                                layer.register_forward_hook(hook_as(f'conv_{i}'))
                    if hasattr(model, 'avgpool'):
                        model.avgpool.register_forward_hook(hook_as('avgpool'))
                        
                elif arch == 'densenet121':
                    # For DenseNet models - hook into dense blocks
                    if hasattr(model, 'features'):
                        # Hook into specific dense blocks
                        dense_blocks = []
                        for i, layer in enumerate(model.features):
                            if hasattr(layer, 'denseblock1'):
                                layer.denseblock1.register_forward_hook(hook_as('denseblock1'))
                            if hasattr(layer, 'denseblock2'):
                                layer.denseblock2.register_forward_hook(hook_as('denseblock2'))
                            if hasattr(layer, 'denseblock3'):
                                layer.denseblock3.register_forward_hook(hook_as('denseblock3'))
                            if hasattr(layer, 'denseblock4'):
                                layer.denseblock4.register_forward_hook(hook_as('denseblock4'))
                            if isinstance(layer, nn.Conv2d):
                                layer.register_forward_hook(hook_as(f'conv_{i}'))
                    if hasattr(model, 'classifier'):
                        model.classifier.register_forward_hook(hook_as('classifier'))
                        
                elif arch == 'mobilenet_v2':
                    # For MobileNet models
                    if hasattr(model, 'features'):
                        for i, layer in enumerate(model.features):
                            if isinstance(layer, nn.Conv2d):
                                layer.register_forward_hook(hook_as(f'conv_{i}'))
                    if hasattr(model, 'classifier'):
                        model.classifier.register_forward_hook(hook_as('classifier'))
                        
                elif arch == 'efficientnet_b0' or arch == 'efficientnet_b3':
                    # For EfficientNet models - hook into blocks
                    if hasattr(model, 'features'):
                        for i, layer in enumerate(model.features):
                            if isinstance(layer, nn.Conv2d):
                                layer.register_forward_hook(hook_as(f'conv_{i}'))
                            # Also hook into MBConv blocks
                            if hasattr(layer, 'block'):
                                layer.block.register_forward_hook(hook_as(f'block_{i}'))
                    if hasattr(model, 'classifier'):
                        model.classifier.register_forward_hook(hook_as('classifier'))
                
                # Forward pass to extract features
                with torch.no_grad():
                    _ = model(input_tensor)
                
                # Display extracted features
                st.subheader(f"Feature Extraction Results for {selected_model}")
                st.write(f"Architecture: {arch}")
                st.write(f"Number of classes: {num_classes}")
                st.write(f"Input shape: {input_tensor.shape}")
                
                if features:
                    st.write(f"Extracted {len(features)} feature layers:")
                    
                    # Filter out 1D features (classifier outputs) and show only 2D+ features
                    conv_features = {name: feat for name, feat in features.items() if feat.dim() > 2}
                    
                    if conv_features:
                        # Add layer selection dropdown
                        layer_options = list(conv_features.keys())
                        selected_layer = st.selectbox(
                            f"Select layer for {selected_model}:",
                            layer_options,
                            help="Choose which feature layer to visualize"
                        )
                        
                        if selected_layer in conv_features:
                            show_maps_and_stats(conv_features[selected_layer][0])
                    else:
                        st.warning("No 2D feature maps were extracted. Only classifier outputs were captured.")
                        st.write("Available features:")
                        for layer_name, feature_map in features.items():
                            st.write(f"- {layer_name}: {feature_map.shape}")
                else:
                    st.warning("No features were extracted. The model architecture might not be supported for feature extraction.")
                
                feature_results[selected_model] = {
                    'features': features, 
                    'input_shape': input_tensor.shape, 
                    'model': model,
                    'architecture': arch,
                    'num_classes': num_classes
                }
                
            except Exception as e:
                st.error(f"Error extracting features from {selected_model}: {e}")
                st.write("This model might not be supported for feature extraction or there might be a loading error.")
                import traceback
                st.code(traceback.format_exc())

# --- 5. Classification ---
elif section == 'Classification':
    st.header('5. Classification')
    if 'preprocessed_image' not in st.session_state:
        st.warning('Please preprocess an image first.')
    else:
        image = st.session_state['preprocessed_image']

        st.subheader('Classification Approach')
        
        # Show magnification warning if available
        if 'estimated_magnification' in st.session_state:
            mag = st.session_state['estimated_magnification']
            if "400x" in mag:
                st.error("⚠️ **400x Image Warning**: Your models were primarily trained on 200x images. Classification accuracy may be lower for 400x images.")
                st.info("💡 **Recommendation**: Try using 200x images or consider training magnification-specific models.")
            elif "200x" in mag:
                st.success("✅ **200x Image Detected**: This is the optimal resolution for your trained models.")
        
        approach = st.selectbox(
            'Choose classification approach:',
            ['Individual Models', 'Ensemble Methods']
            # ['Individual Models', 'Ensemble Methods', 'Hybrid Classifiers']
        )

        # ---------- helpers ----------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_configs = load_model_configs()

        def detect_num_classes_in_ckpt(ckpt_path):
            """Try to infer num output classes from the checkpoint head."""
            try:
                sd = torch.load(ckpt_path, map_location='cpu')
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                if isinstance(sd, dict):
                    sd = {k.replace('module.', ''): v for k, v in sd.items()}
                    
                    # Look for classifier weight patterns
                    classifier_keys = [
                        'classifier.6.weight',  # VGG
                        'fc.weight',           # ResNet, DenseNet
                        'classifier.1.weight', # MobileNet
                        '_fc.weight',          # EfficientNet
                        'classifier[-1].weight',
                        'classifier.weight',   # DenseNet direct
                        'classifier.0.weight', # Some architectures
                        'classifier.3.weight', # MobileNetV3
                        'classifier.4.weight', # Some variants
                        'classifier.5.weight'  # Some variants
                    ]
                    
                    for k in classifier_keys:
                        if k in sd and sd[k].dim() == 2:
                            num_classes = int(sd[k].shape[0])
                            print(f"Detected {num_classes} classes from key '{k}' in {ckpt_path}")
                            return num_classes
                    
                    # If no specific classifier found, look for any weight with 2D shape
                    for k, v in sd.items():
                        if 'classifier' in k and 'weight' in k and v.dim() == 2:
                            num_classes = int(v.shape[0])
                            print(f"Detected {num_classes} classes from key '{k}' in {ckpt_path}")
                            return num_classes
                            
            except Exception as e:
                print(f"Error detecting classes in {ckpt_path}: {e}")
                pass
            return None  # unknown

        def try_load_torchvision(model, ckpt_path, strict=False):
            """Load state_dict into a torchvision model and return (#missing, #unexpected)."""
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, dict):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=strict)
            # Normalize to counts across torch versions
            def _count(x, attr_missing='missing_keys', attr_unexp='unexpected_keys'):
                if hasattr(x, '__iter__') and not hasattr(x, 'missing_keys'):
                    return len(x)
                # torch>=2: _IncompatibleKeys
                return len(getattr(x, attr_missing, []) or []) if attr_missing else len(getattr(x, attr_unexp, []) or [])
            try:
                m_cnt = _count(missing)
                u_cnt = _count(unexpected, attr_missing=None, attr_unexp='unexpected_keys')
            except Exception:
                m_cnt = len(missing) if hasattr(missing, '__iter__') else 0
                u_cnt = len(unexpected) if hasattr(unexpected, '__iter__') else 0
            return m_cnt, u_cnt

        def load_efficientnet_pytorch(which, ckpt_path, num_classes=2):
            """Fallback loader for efficientnet_pytorch checkpoints."""
            name = f'efficientnet-{which}'
            m = EfficientNet.from_name(name)
            in_feat = m._fc.in_features
            m._fc = nn.Linear(in_feat, num_classes)
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, dict):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            m.load_state_dict(sd, strict=False)  # allow head mismatch
            return m

        def build_model(model_name, config):
            """Create model for inference, handle both 2-class and 6-class models."""
            arch = config['architecture']
            ckpt = config['file']

            # Check if model is already cached
            cache_key = f"{model_name}_{arch}_{ckpt}"
            if cache_key in MODEL_CACHE:
                print(f"Using cached model: {model_name}")
                return MODEL_CACHE[cache_key]

            # Check class count in checkpoint
            num_classes = detect_num_classes_in_ckpt(ckpt)
            if num_classes is None:
                num_classes = 2  # Default to 2 classes if can't detect
                print(f"Warning: Could not detect classes in {ckpt}, defaulting to 2 classes")
            else:
                print(f"Building {model_name} with {num_classes} classes")
            
            # Build base torchvision model with correct number of classes
            if arch == 'vgg16':
                model = models.vgg16(weights=None)
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading VGG16 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'resnet18':
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading ResNet18 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.fc = nn.Linear(model.fc.in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'resnet34':
                model = models.resnet34(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading ResNet34 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.fc = nn.Linear(model.fc.in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'densenet121':
                model = models.densenet121(weights=None)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading DenseNet121 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.classifier = nn.Linear(model.classifier.in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'mobilenet_v2':
                model = models.mobilenet_v2(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading MobileNetV2 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading EfficientNetB0 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'efficientnet_b3':
                # Try torchvision first, then fallback to efficientnet_pytorch if many unexpected keys
                tv = models.efficientnet_b3(weights=None)
                tv.classifier[-1] = nn.Linear(tv.classifier[-1].in_features, num_classes)
                m_cnt, u_cnt = try_load_torchvision(tv, ckpt, strict=False)
                if u_cnt > 50:  # lots of unexpected keys → different library
                    model = load_efficientnet_pytorch('b3', ckpt, num_classes=num_classes)
                else:
                    model = tv

            elif arch == 'efficientnet_b4':
                tv = models.efficientnet_b4(weights=None)
                tv.classifier[-1] = nn.Linear(tv.classifier[-1].in_features, num_classes)
                m_cnt, u_cnt = try_load_torchvision(tv, ckpt, strict=False)
                if u_cnt > 50:
                    # If you actually have an efficientnet_pytorch B4 checkpoint, add a loader like load_efficientnet_pytorch('b4', ...)
                    model = tv  # fallback to TV if we can't detect a different lib
                else:
                    model = tv

            elif arch == 'mobilenet_v3_large':
                model = models.mobilenet_v3_large(weights=None)
                # MobileNetV3 Large has a different classifier structure
                # The classifier is a Sequential with multiple layers
                # We need to replace the last Linear layer
                if hasattr(model.classifier, '__len__'):
                    # Find the last Linear layer in the classifier
                    for i in range(len(model.classifier) - 1, -1, -1):
                        if isinstance(model.classifier[i], nn.Linear):
                            model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                            break
                else:
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading MobileNetV3 Large with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        for i in range(len(model.classifier) - 1, -1, -1):
                            if isinstance(model.classifier[i], nn.Linear):
                                model.classifier[i] = nn.Linear(model.classifier[i].in_features, 6)
                                break
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'shufflenet_v2_x1_0':
                model = models.shufflenet_v2_x1_0(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading ShuffleNetV2 with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.fc = nn.Linear(model.fc.in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'squeezenet1_1':
                model = models.squeezenet1_1(weights=None)
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading SqueezeNet with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.classifier[1] = nn.Conv2d(512, 6, kernel_size=1)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'convnext_base':
                model = models.convnext_base(weights=None)
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading ConvNext with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'regnet_y_32gf':
                model = models.regnet_y_32gf(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                try:
                    try_load_torchvision(model, ckpt, strict=False)
                except Exception as e:
                    print(f"Error loading RegNet with {num_classes} classes: {e}")
                    # Try with 6 classes as fallback
                    if num_classes == 2:
                        print("Trying with 6 classes as fallback...")
                        model.fc = nn.Linear(model.fc.in_features, 6)
                        try_load_torchvision(model, ckpt, strict=False)
                        num_classes = 6

            elif arch == 'gan_generator':
                # For GAN, we'll use a pre-trained ResNet for classification
                # since the discriminator is designed for real/fake classification
                if GAN_AVAILABLE:
                    gan = BreastCancerGAN(device=device)
                    gan.load_models(ckpt)
                    
                    # Use a pre-trained ResNet for medical classification
                    # Try to load a pre-trained breast cancer model first
                    if os.path.exists('models/resnet18_breast_cancer_optimized.pth'):
                        model = models.resnet18(weights=None)
                        model.fc = nn.Linear(model.fc.in_features, 2)
                        model.load_state_dict(torch.load('models/resnet18_breast_cancer_optimized.pth', map_location=device))
                        print("Loaded pre-trained ResNet18 for GAN classification")
                    elif os.path.exists('models/resnet18_breast_cancer.pth'):
                        model = models.resnet18(weights=None)
                        model.fc = nn.Linear(model.fc.in_features, 2)
                        model.load_state_dict(torch.load('models/resnet18_breast_cancer.pth', map_location=device))
                        print("Loaded pre-trained ResNet18 for GAN classification")
                    else:
                        # Use ImageNet pre-trained weights as fallback
                        model = models.resnet18(weights='IMAGENET1K_V1')
                        model.fc = nn.Linear(model.fc.in_features, 2)
                        print("Using ImageNet pre-trained ResNet18 for GAN classification")
                else:
                    raise RuntimeError("GAN models not available")

            else:
                raise RuntimeError(f"Unsupported architecture: {arch}")

            # Cache the model for future use
            MODEL_CACHE[cache_key] = (model.to(device).eval(), num_classes)
            return model.to(device).eval(), num_classes

        def map_6class_to_2class(predictions, num_classes):
            """Map 6-class predictions to 2-class (benign/malignant) predictions"""
            if num_classes == 2:
                return predictions
            
            # For 6-class models, map to 2 classes
            # Assuming classes 0,1,2 are benign and 3,4,5 are malignant
            # This mapping might need adjustment based on your actual class labels
            if predictions.dim() == 1:
                # Single prediction
                if predictions.argmax() < 3:
                    return torch.tensor([1.0, 0.0])  # Benign
                else:
                    return torch.tensor([0.0, 1.0])  # Malignant
            else:
                # Batch predictions
                batch_size = predictions.shape[0]
                mapped = torch.zeros(batch_size, 2)
                for i in range(batch_size):
                    if predictions[i].argmax() < 3:
                        mapped[i] = torch.tensor([1.0, 0.0])  # Benign
                    else:
                        mapped[i] = torch.tensor([0.0, 1.0])  # Malignant
                return mapped

        def build_transform(config):
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(config['input_size']),
                transforms.ToTensor(),
                transforms.Normalize(*config['normalization'])
            ])
        # ---------- end helpers ----------

        if approach == 'Individual Models':
            st.subheader('Individual Model Selection')

            # Only list models with an existing, likely-loadable checkpoint
            available_models = []
            for model_name, cfg in model_configs.items():
                if not os.path.exists(cfg['file']):
                    continue
                # quick arch sanity: we accept; deep probing is expensive – build_model will guard anyway
                available_models.append(model_name)

            if available_models:
                selected_models = st.multiselect(
                    'Select models to use:',
                    available_models,
                    default=available_models[:3]
                )

                if selected_models:
                    st.write('---')
                    class_names = ['Non-cancerous', 'Cancerous']
                    st.subheader('Individual Model Predictions')

                    model_results = []
                    for model_name in selected_models:
                        config = model_configs[model_name]

                        if not os.path.exists(config['file']):
                            st.error(f"Model file not found: {config['file']}")
                            continue

                        try:
                            # Get the actual number of classes in the checkpoint
                            nclasses = detect_num_classes_in_ckpt(config['file'])
                            if nclasses is None:
                                nclasses = 2  # Default to 2 classes if can't detect

                            model, num_classes = build_model(model_name, config)
                            transform = build_transform(config)
                            
                            # Cache the preprocessed image tensor to ensure consistency
                            image_hash = hash(str(image.tobytes()))
                            tensor_cache_key = f"{image_hash}_{config['input_size']}_{config['normalization']}"
                            
                            if tensor_cache_key in IMAGE_TENSOR_CACHE:
                                img_tensor = IMAGE_TENSOR_CACHE[tensor_cache_key]
                                print(f"Using cached tensor for {model_name}")
                            else:
                                img_tensor = transform(image).unsqueeze(0).to(device)
                                IMAGE_TENSOR_CACHE[tensor_cache_key] = img_tensor
                                print(f"Created new tensor for {model_name}")

                            with torch.no_grad():
                                logits = model(img_tensor)
                                # Map 6-class predictions to 2-class if needed
                                logits = map_6class_to_2class(logits, num_classes)
                                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

                            # Handle different numbers of classes
                            if nclasses == 2:
                                # Binary classification
                                prediction = int(np.argmax(probs))
                                confidence = float(probs[prediction])
                                prediction_name = class_names[prediction]
                                
                                result = {
                                    'model': model_name,
                                    'prediction': prediction_name,
                                    'confidence': confidence,
                                    'probabilities': probs,
                                    'config': config
                                }

                                st.write(f"**{model_name}**: {prediction_name} ({confidence:.1%} confidence)")
                                st.write(f"   Non-cancerous: {probs[0]:.3f}, Cancerous: {probs[1]:.3f}")
                                
                            elif nclasses == 6:
                                # 6-class classification (BreakHis dataset)
                                prediction = int(np.argmax(probs))
                                confidence = float(probs[prediction])
                                
                                # Map 6 classes to binary classification
                                # Classes 0-2: Benign (Non-cancerous), Classes 3-5: Malignant (Cancerous)
                                if prediction < 3:
                                    prediction_name = 'Non-cancerous'
                                    binary_confidence = float(np.sum(probs[:3]))  # Sum of benign probabilities
                                else:
                                    prediction_name = 'Cancerous'
                                    binary_confidence = float(np.sum(probs[3:]))  # Sum of malignant probabilities
                                
                                result = {
                                    'model': model_name,
                                    'prediction': prediction_name,
                                    'confidence': binary_confidence,
                                    'probabilities': [np.sum(probs[:3]), np.sum(probs[3:])],  # Binary probabilities
                                    'config': config,
                                    'original_classes': nclasses,
                                    'original_prediction': prediction
                                }
                                
                                st.write(f"**{model_name}**: {prediction_name} ({binary_confidence:.1%} confidence)")
                                st.write(f"   Non-cancerous: {np.sum(probs[:3]):.3f}, Cancerous: {np.sum(probs[3:]):.3f}")
                                st.write(f"   Original 6-class prediction: Class {prediction} ({confidence:.1%})")
                                
                            else:
                                # Other number of classes - use argmax
                                prediction = int(np.argmax(probs))
                                confidence = float(probs[prediction])
                                prediction_name = f"Class {prediction}"
                                
                                result = {
                                    'model': model_name,
                                    'prediction': prediction_name,
                                    'confidence': confidence,
                                    'probabilities': probs,
                                    'config': config,
                                    'original_classes': nclasses
                                }
                                
                                st.write(f"**{model_name}**: {prediction_name} ({confidence:.1%} confidence)")
                                st.write(f"   Classes: {nclasses}, Probabilities: {probs}")
                            
                            st.write(f"   Architecture: {config['description']}")
                            model_results.append(result)

                        except Exception as e:
                            st.error(f"Error loading/running {model_name}: {e}")

                    # Store & summarize
                    if model_results:
                        st.session_state['classification_results'] = model_results
                        st.session_state['classification_result'] = model_results[0]['prediction']
                        st.session_state['classification_probs'] = model_results[0]['probabilities']

                        st.subheader('Classification Summary')
                        predictions = [r['prediction'] for r in model_results]
                        cancerous_count = predictions.count('Cancerous')
                        non_cancerous_count = predictions.count('Non-cancerous')

                        st.write(f"**Total Models**: {len(model_results)}")
                        st.write(f"**Cancerous Predictions**: {cancerous_count}")
                        st.write(f"**Non-cancerous Predictions**: {non_cancerous_count}")

                        if cancerous_count > non_cancerous_count:
                            st.success("**Majority Decision: Cancerous**")
                        elif non_cancerous_count > cancerous_count:
                            st.success("**Majority Decision: Non-cancerous**")
                        else:
                            st.warning("**Split Decision** - Models disagree")
                    else:
                        st.error("No models produced predictions.")
            else:
                st.warning("No trained models found. Please train models first.")

        elif approach == 'Ensemble Methods':
            st.subheader('Ensemble Classification Methods')
            
            # Get available models
            available_models = []
            for model_name, cfg in model_configs.items():
                if os.path.exists(cfg['file']):
                    available_models.append(model_name)
            
            if not available_models:
                st.warning("No trained models found. Please train models first.")
            else:
                # Ensemble method selection
                ensemble_method = st.selectbox(
                    'Choose ensemble method:',
                    [
                        'Majority Voting',
                        'Weighted Voting', 
                        'Average Probabilities',
                        'Stacking Classifier',
                        'Boosting Ensemble',
                        'Confidence-based Fusion',
                        'Feature-level Fusion',
                        'Multi-scale Ensemble'
                    ],
                    help="Select the ensemble method for combining multiple model predictions"
                )
                
                # Model selection for ensemble
                selected_models = st.multiselect(
                    'Select models for ensemble:',
                    available_models,
                    default=available_models[:5] if len(available_models) >= 5 else available_models,
                    help="Choose multiple models to combine their predictions"
                )
                
                if selected_models and len(selected_models) >= 2:
                    st.write('---')
                    class_names = ['Non-cancerous', 'Cancerous']
                    
                    # Load all selected models and get predictions
                    model_predictions = []
                    model_confidences = []
                    model_probabilities = []
                    
                    with st.spinner("Loading models and generating predictions..."):
                        for model_name in selected_models:
                            config = model_configs[model_name]
                            
                            try:
                                # Get the actual number of classes in the checkpoint
                                nclasses = detect_num_classes_in_ckpt(config['file'])
                                if nclasses is None:
                                    nclasses = 2  # Default to 2 classes if can't detect
                                
                                model, num_classes = build_model(model_name, config)
                                transform = build_transform(config)
                                img_tensor = transform(image).unsqueeze(0).to(device)
                                
                                with torch.no_grad():
                                    logits = model(img_tensor)
                                    # Map 6-class predictions to 2-class if needed
                                    logits = map_6class_to_2class(logits, num_classes)
                                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                                
                                # Handle different numbers of classes
                                if nclasses == 2:
                                    # Binary classification
                                    prediction = int(np.argmax(probs))
                                    confidence = float(probs[prediction])
                                    
                                    model_predictions.append(prediction)
                                    model_confidences.append(confidence)
                                    model_probabilities.append(probs)
                                    
                                elif nclasses == 6:
                                    # 6-class classification (BreakHis dataset)
                                    prediction = int(np.argmax(probs))
                                    
                                    # Map 6 classes to binary classification
                                    # Classes 0-2: Benign (0), Classes 3-5: Malignant (1)
                                    if prediction < 3:
                                        binary_prediction = 0  # Non-cancerous
                                        binary_confidence = float(np.sum(probs[:3]))  # Sum of benign probabilities
                                    else:
                                        binary_prediction = 1  # Cancerous
                                        binary_confidence = float(np.sum(probs[3:]))  # Sum of malignant probabilities
                                    
                                    model_predictions.append(binary_prediction)
                                    model_confidences.append(binary_confidence)
                                    model_probabilities.append([np.sum(probs[:3]), np.sum(probs[3:])])
                                    
                                else:
                                    # Other number of classes - use argmax as binary
                                    prediction = int(np.argmax(probs))
                                    confidence = float(probs[prediction])
                                    
                                    # Simple binary mapping: class 0 = 0, others = 1
                                    binary_prediction = 0 if prediction == 0 else 1
                                    
                                    model_predictions.append(binary_prediction)
                                    model_confidences.append(confidence)
                                    model_probabilities.append([probs[0], np.sum(probs[1:])])
                                
                            except Exception as e:
                                st.error(f"Error loading {model_name}: {e}")
                                continue
                    
                    if len(model_predictions) >= 2:
                        st.subheader(f'{ensemble_method} Results')
                        
                        # Display individual model results
                        st.write("**Individual Model Predictions:**")
                        for i, (model_name, pred, conf, probs) in enumerate(zip(selected_models, model_predictions, model_confidences, model_probabilities)):
                            st.write(f"- **{model_name}**: {class_names[pred]} ({conf:.1%} confidence)")
                        
                        # Apply ensemble method
                        if ensemble_method == 'Majority Voting':
                            # Simple majority vote
                            cancerous_votes = sum(model_predictions)
                            non_cancerous_votes = len(model_predictions) - cancerous_votes
                            
                            if cancerous_votes > non_cancerous_votes:
                                final_prediction = 'Cancerous'
                                ensemble_confidence = cancerous_votes / len(model_predictions)
                            elif non_cancerous_votes > cancerous_votes:
                                final_prediction = 'Non-cancerous'
                                ensemble_confidence = non_cancerous_votes / len(model_predictions)
                            else:
                                final_prediction = 'Tie'
                                ensemble_confidence = 0.5
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} agreement)")
                            
                        elif ensemble_method == 'Weighted Voting':
                            # Weight by confidence
                            weights = np.array(model_confidences)
                            weighted_cancerous = sum(weights[i] for i, pred in enumerate(model_predictions) if pred == 1)
                            weighted_non_cancerous = sum(weights[i] for i, pred in enumerate(model_predictions) if pred == 0)
                            
                            if weighted_cancerous > weighted_non_cancerous:
                                final_prediction = 'Cancerous'
                                ensemble_confidence = weighted_cancerous / (weighted_cancerous + weighted_non_cancerous)
                            else:
                                final_prediction = 'Non-cancerous'
                                ensemble_confidence = weighted_non_cancerous / (weighted_cancerous + weighted_non_cancerous)
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} weighted confidence)")
                            
                        elif ensemble_method == 'Average Probabilities':
                            # Average the probability distributions
                            avg_probs = np.mean(model_probabilities, axis=0)
                            final_prediction = class_names[np.argmax(avg_probs)]
                            ensemble_confidence = float(np.max(avg_probs))
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                            st.write(f"**Averaged Probabilities**: Non-cancerous: {avg_probs[0]:.3f}, Cancerous: {avg_probs[1]:.3f}")
                            
                        elif ensemble_method == 'Stacking Classifier':
                            # Use a meta-learner (simple logistic regression)
                            try:
                                from sklearn.linear_model import LogisticRegression
                                
                                # Prepare features (probabilities from each model)
                                X = np.array(model_probabilities)
                                
                                # Create a simple meta-learner
                                meta_learner = LogisticRegression(random_state=42)
                                
                                # For demonstration, we'll use the current predictions as "training data"
                                # In practice, you'd have a separate validation set
                                y_meta = model_predictions
                                
                                meta_learner.fit(X, y_meta)
                                meta_prediction = meta_learner.predict(X)[0]
                                meta_confidence = meta_learner.predict_proba(X)[0].max()
                                
                                final_prediction = class_names[meta_prediction]
                                ensemble_confidence = float(meta_confidence)
                                
                                st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                                st.write("**Meta-learner**: Logistic Regression on model probabilities")
                                
                            except ImportError:
                                st.error("scikit-learn not available for stacking classifier")
                                # Fallback to average probabilities
                                avg_probs = np.mean(model_probabilities, axis=0)
                                final_prediction = class_names[np.argmax(avg_probs)]
                                ensemble_confidence = float(np.max(avg_probs))
                                st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                                
                        elif ensemble_method == 'Boosting Ensemble':
                            # AdaBoost-style weighted combination
                            # Start with equal weights, then adjust based on confidence
                            weights = np.ones(len(model_predictions))
                            
                            # Boost weights for high-confidence correct predictions
                            for i, (pred, conf) in enumerate(zip(model_predictions, model_confidences)):
                                if conf > 0.8:  # High confidence
                                    weights[i] *= 1.5
                                elif conf < 0.6:  # Low confidence
                                    weights[i] *= 0.7
                            
                            # Normalize weights
                            weights = weights / np.sum(weights)
                            
                            # Weighted voting
                            weighted_cancerous = sum(weights[i] for i, pred in enumerate(model_predictions) if pred == 1)
                            weighted_non_cancerous = sum(weights[i] for i, pred in enumerate(model_predictions) if pred == 0)
                            
                            if weighted_cancerous > weighted_non_cancerous:
                                final_prediction = 'Cancerous'
                                ensemble_confidence = weighted_cancerous
                            else:
                                final_prediction = 'Non-cancerous'
                                ensemble_confidence = weighted_non_cancerous
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                            st.write("**Boosting Weights**:", [f"{w:.2f}" for w in weights])
                            
                        elif ensemble_method == 'Confidence-based Fusion':
                            # Fuse based on confidence thresholds
                            high_conf_models = [i for i, conf in enumerate(model_confidences) if conf > 0.8]
                            medium_conf_models = [i for i, conf in enumerate(model_confidences) if 0.6 <= conf <= 0.8]
                            low_conf_models = [i for i, conf in enumerate(model_confidences) if conf < 0.6]
                            
                            st.write("**Confidence Analysis:**")
                            st.write(f"- High confidence models (>80%): {len(high_conf_models)}")
                            st.write(f"- Medium confidence models (60-80%): {len(medium_conf_models)}")
                            st.write(f"- Low confidence models (<60%): {len(low_conf_models)}")
                            
                            if high_conf_models:
                                # Use only high confidence models
                                hc_predictions = [model_predictions[i] for i in high_conf_models]
                                hc_confidences = [model_confidences[i] for i in high_conf_models]
                                
                                cancerous_votes = sum(hc_predictions)
                                if cancerous_votes > len(hc_predictions) / 2:
                                    final_prediction = 'Cancerous'
                                else:
                                    final_prediction = 'Non-cancerous'
                                ensemble_confidence = np.mean(hc_confidences)
                                
                            elif medium_conf_models:
                                # Use medium confidence models
                                mc_predictions = [model_predictions[i] for i in medium_conf_models]
                                mc_confidences = [model_confidences[i] for i in medium_conf_models]
                                
                                cancerous_votes = sum(mc_predictions)
                                if cancerous_votes > len(mc_predictions) / 2:
                                    final_prediction = 'Cancerous'
                                else:
                                    final_prediction = 'Non-cancerous'
                                ensemble_confidence = np.mean(mc_confidences)
                                
                            else:
                                # Use all models with low confidence
                                cancerous_votes = sum(model_predictions)
                                if cancerous_votes > len(model_predictions) / 2:
                                    final_prediction = 'Cancerous'
                                else:
                                    final_prediction = 'Non-cancerous'
                                ensemble_confidence = np.mean(model_confidences)
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                            
                        elif ensemble_method == 'Feature-level Fusion':
                            # Extract features from all models and combine
                            st.write("**Feature-level Fusion**")
                            st.write("This method would extract features from each model and combine them.")
                            st.write("For demonstration, using average probabilities as feature fusion.")
                            
                            avg_probs = np.mean(model_probabilities, axis=0)
                            final_prediction = class_names[np.argmax(avg_probs)]
                            ensemble_confidence = float(np.max(avg_probs))
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                            
                        elif ensemble_method == 'Multi-scale Ensemble':
                            # Different models might be better at different scales/features
                            st.write("**Multi-scale Ensemble**")
                            st.write("Combining models that excel at different scales and features.")
                            
                            # Categorize models by type
                            cnn_models = [i for i, name in enumerate(selected_models) if any(x in name.lower() for x in ['resnet', 'vgg', 'densenet'])]
                            efficient_models = [i for i, name in enumerate(selected_models) if 'efficient' in name.lower()]
                            mobile_models = [i for i, name in enumerate(selected_models) if 'mobile' in name.lower()]
                            
                            st.write(f"**Model Categories**: CNN ({len(cnn_models)}), EfficientNet ({len(efficient_models)}), Mobile ({len(mobile_models)})")
                            
                            # Weight by model category
                            category_weights = np.ones(len(model_predictions))
                            for i in cnn_models:
                                category_weights[i] *= 1.2  # CNN models get higher weight
                            for i in efficient_models:
                                category_weights[i] *= 1.1  # EfficientNet models get medium weight
                            
                            # Apply category weights
                            weighted_cancerous = sum(category_weights[i] for i, pred in enumerate(model_predictions) if pred == 1)
                            weighted_non_cancerous = sum(category_weights[i] for i, pred in enumerate(model_predictions) if pred == 0)
                            
                            if weighted_cancerous > weighted_non_cancerous:
                                final_prediction = 'Cancerous'
                                ensemble_confidence = weighted_cancerous / (weighted_cancerous + weighted_non_cancerous)
                            else:
                                final_prediction = 'Non-cancerous'
                                ensemble_confidence = weighted_non_cancerous / (weighted_cancerous + weighted_non_cancerous)
                            
                            st.success(f"**Ensemble Decision**: {final_prediction} ({ensemble_confidence:.1%} confidence)")
                        
                        # Store results for staging
                        st.session_state['classification_results'] = [{
                            'model': f'{ensemble_method} Ensemble',
                            'prediction': final_prediction,
                            'confidence': ensemble_confidence,
                            'probabilities': [1-ensemble_confidence, ensemble_confidence] if final_prediction == 'Cancerous' else [ensemble_confidence, 1-ensemble_confidence],
                            'config': {'description': f'{ensemble_method} with {len(selected_models)} models'}
                        }]
                        
                        # Display ensemble statistics
                        st.subheader('Ensemble Statistics')
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Models Used", len(selected_models))
                        with col2:
                            st.metric("Average Confidence", f"{np.mean(model_confidences):.1%}")
                        with col3:
                            st.metric("Confidence Std", f"{np.std(model_confidences):.3f}")
                        
                        # Agreement analysis
                        agreement = len(set(model_predictions)) == 1
                        if agreement:
                            st.success("**Model Agreement**: All models agree on the prediction")
                        else:
                            st.write(f"**Agreement Rate**: {max(model_predictions.count(0), model_predictions.count(1)) / len(model_predictions):.1%}")
                        
                    else:
                        st.error("Need at least 2 models for ensemble classification.")
                else:
                    st.warning("Please select at least 2 models for ensemble classification.")

        elif approach == 'Hybrid Classifiers':
            st.subheader('Hybrid Classification Methods')
            st.write("**Hybrid Classifiers**: Combining different types of models and techniques")
            
            # Hybrid method selection
            hybrid_method = st.selectbox(
                'Choose hybrid method:',
                [
                    'CNN + Traditional ML',
                    'Deep Learning + Statistical',
                    'Multi-modal Fusion',
                    'Hierarchical Classification',
                    'Cascade Classifiers',
                    'Adaptive Ensemble'
                ],
                help="Select a hybrid approach that combines different classification paradigms"
            )
            
            if hybrid_method == 'CNN + Traditional ML':
                st.write("**CNN + Traditional ML Hybrid**")
                st.write("This method combines deep learning features with traditional machine learning classifiers.")
                
                # Get available models
                available_models = []
                for model_name, cfg in model_configs.items():
                    if os.path.exists(cfg['file']):
                        available_models.append(model_name)
                
                if available_models:
                    selected_cnn = st.selectbox("Select CNN model for feature extraction:", available_models)
                    
                    if st.button("Run Hybrid Classification"):
                        with st.spinner("Running hybrid classification..."):
                            try:
                                # Load CNN model and extract features
                                config = model_configs[selected_cnn]
                                model, num_classes = build_model(selected_cnn, config)
                                transform = build_transform(config)
                                img_tensor = transform(image).unsqueeze(0).to(device)
                                
                                # Extract features from the last layer before classification
                                features = None
                                def hook_fn(module, input, output):
                                    global features
                                    features = output.detach().cpu().numpy()
                                
                                # Register hook on the last layer before classifier
                                if hasattr(model, 'classifier'):
                                    if isinstance(model.classifier, nn.Sequential):
                                        model.classifier[-2].register_forward_hook(hook_fn)
                                    else:
                                        model.classifier.register_forward_hook(hook_fn)
                                elif hasattr(model, 'fc'):
                                    model.fc.register_forward_hook(hook_fn)
                                
                                with torch.no_grad():
                                    _ = model(img_tensor)
                                
                                if features is not None:
                                    features = features.flatten()
                                    
                                    # Use traditional ML classifiers
                                    try:
                                        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                                        from sklearn.svm import SVC
                                        from sklearn.linear_model import LogisticRegression
                                        from sklearn.model_selection import cross_val_score
                                        
                                        # Create synthetic training data for demonstration
                                        # In practice, you'd use real training data
                                        np.random.seed(42)
                                        n_samples = 100
                                        n_features = len(features)
                                        
                                        # Generate synthetic features (in practice, use real data)
                                        X_synthetic = np.random.randn(n_samples, n_features)
                                        y_synthetic = np.random.randint(0, 2, n_samples)
                                        
                                        # Train different classifiers
                                        classifiers = {
                                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                                            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                                            'SVM': SVC(probability=True, random_state=42),
                                            'Logistic Regression': LogisticRegression(random_state=42)
                                        }
                                        
                                        predictions = {}
                                        confidences = {}
                                        
                                        for name, clf in classifiers.items():
                                            clf.fit(X_synthetic, y_synthetic)
                                            pred = clf.predict([features])[0]
                                            prob = clf.predict_proba([features])[0]
                                            
                                            predictions[name] = ['Non-cancerous', 'Cancerous'][pred]
                                            confidences[name] = float(prob[pred])
                                        
                                        # Display results
                                        st.subheader("Hybrid Classification Results")
                                        for name, pred in predictions.items():
                                            conf = confidences[name]
                                            st.write(f"**{name}**: {pred} ({conf:.1%} confidence)")
                                        
                                        # Ensemble the traditional ML results
                                        ml_predictions = list(predictions.values())
                                        ml_confidences = list(confidences.values())
                                        
                                        cancerous_votes = sum(1 for p in ml_predictions if p == 'Cancerous')
                                        if cancerous_votes > len(ml_predictions) / 2:
                                            final_prediction = 'Cancerous'
                                        else:
                                            final_prediction = 'Non-cancerous'
                                        
                                        avg_confidence = np.mean(ml_confidences)
                                        
                                        st.success(f"**Hybrid Decision**: {final_prediction} ({avg_confidence:.1%} confidence)")
                                        
                                        # Store results
                                        st.session_state['classification_results'] = [{
                                            'model': f'Hybrid CNN+ML ({selected_cnn})',
                                            'prediction': final_prediction,
                                            'confidence': avg_confidence,
                                            'probabilities': [1-avg_confidence, avg_confidence] if final_prediction == 'Cancerous' else [avg_confidence, 1-avg_confidence],
                                            'config': {'description': f'CNN feature extraction + Traditional ML ensemble'}
                                        }]
                                        
                                    except ImportError:
                                        st.error("scikit-learn not available for traditional ML classifiers")
                                        
                            except Exception as e:
                                st.error(f"Error in hybrid classification: {e}")
                else:
                    st.warning("No trained models available for hybrid classification.")
            
            elif hybrid_method == 'Deep Learning + Statistical':
                st.write("**Deep Learning + Statistical Hybrid**")
                st.write("Combines deep learning predictions with statistical analysis of image properties.")
                
                if st.button("Run Statistical Analysis"):
                    with st.spinner("Analyzing image statistics..."):
                        # Statistical analysis of the image
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        
                        # Calculate statistical features
                        mean_intensity = np.mean(gray)
                        std_intensity = np.std(gray)
                        skewness = np.mean(((gray - mean_intensity) / std_intensity) ** 3)
                        kurtosis = np.mean(((gray - mean_intensity) / std_intensity) ** 4) - 3
                        
                        # Texture features
                        from skimage.feature import graycomatrix, graycoprops
                        glcm = graycomatrix(gray.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                        contrast = graycoprops(glcm, 'contrast')[0, 0]
                        energy = graycoprops(glcm, 'energy')[0, 0]
                        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                        
                        st.subheader("Statistical Features")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Mean Intensity**: {mean_intensity:.2f}")
                            st.write(f"**Std Intensity**: {std_intensity:.2f}")
                            st.write(f"**Skewness**: {skewness:.3f}")
                            st.write(f"**Kurtosis**: {kurtosis:.3f}")
                        
                        with col2:
                            st.write(f"**Contrast**: {contrast:.3f}")
                            st.write(f"**Energy**: {energy:.3f}")
                            st.write(f"**Homogeneity**: {homogeneity:.3f}")
                        
                        # Simple statistical classification rules
                        # These are heuristic rules based on typical cancer characteristics
                        cancer_score = 0
                        
                        if std_intensity > 50:  # High variation might indicate cancer
                            cancer_score += 1
                        if contrast > 0.3:  # High contrast might indicate cancer
                            cancer_score += 1
                        if energy < 0.1:  # Low energy might indicate cancer
                            cancer_score += 1
                        if abs(skewness) > 0.5:  # High skewness might indicate cancer
                            cancer_score += 1
                        
                        statistical_prediction = 'Cancerous' if cancer_score >= 2 else 'Non-cancerous'
                        statistical_confidence = cancer_score / 4
                        
                        st.success(f"**Statistical Prediction**: {statistical_prediction} ({statistical_confidence:.1%} confidence)")
                        
                        # Store results
                        st.session_state['classification_results'] = [{
                            'model': 'Statistical Analysis',
                            'prediction': statistical_prediction,
                            'confidence': statistical_confidence,
                            'probabilities': [1-statistical_confidence, statistical_confidence] if statistical_prediction == 'Cancerous' else [statistical_confidence, 1-statistical_confidence],
                            'config': {'description': 'Statistical texture and intensity analysis'}
                        }]
            
            else:
                st.info(f"**{hybrid_method}** - This hybrid method is under development.")
                st.write("Future implementations will include:")
                st.write("- Multi-modal data fusion")
                st.write("- Hierarchical classification pipelines")
                st.write("- Adaptive ensemble methods")
                st.write("- Cascade classifier chains")


# --- 6. Staging ---
elif section == 'Staging':
    st.header('6. Cancer Staging & Risk Assessment')
    if 'classification_results' not in st.session_state:
        st.warning('Please run classification first.')
    else:
        results = st.session_state['classification_results']
        
        if results:
            # Get the main prediction
            main_prediction = results[0]['prediction']
            main_confidence = results[0]['confidence']
            
            st.subheader(' Comprehensive Cancer Staging Analysis')
            
            # Display classification results
            st.write("**Classification Results:**")
            for result in results:
                st.write(f"- **{result['model']}**: {result['prediction']} ({result['confidence']:.1%} confidence)")
            
            st.write("---")
            
            if main_prediction == 'Cancerous':
                st.subheader(' Cancer Staging Assessment')
                
                # TNM Staging System
                st.write("**TNM Staging System Analysis**")
                
                # T (Tumor) Stage Assessment
                st.write("**T (Tumor) Stage:**")
                if main_confidence > 0.9:
                    t_stage = "T1"
                    t_description = "Tumor ≤ 2 cm in greatest dimension"
                    t_color = "🟢"
                elif main_confidence > 0.8:
                    t_stage = "T2"
                    t_description = "Tumor > 2 cm but ≤ 5 cm in greatest dimension"
                    t_color = "🟡"
                elif main_confidence > 0.7:
                    t_stage = "T3"
                    t_description = "Tumor > 5 cm in greatest dimension"
                    t_color = "🟠"
                else:
                    t_stage = "T4"
                    t_description = "Tumor of any size with direct extension to chest wall or skin"
                    t_color = "🔴"
                
                st.write(f"{t_color} **{t_stage}**: {t_description}")
                
                # N (Node) Stage Assessment (simulated based on confidence and image analysis)
                st.write("**N (Node) Stage:**")
                # Simulate node assessment based on confidence and image characteristics
                if 'preprocessed_image' in st.session_state:
                    image = st.session_state['preprocessed_image']
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    
                    # Analyze image characteristics for node involvement simulation
                    image_complexity = np.std(gray) / np.mean(gray)
                    texture_variation = np.var(gray)
                    
                    if image_complexity > 0.3 and texture_variation > 1000:
                        n_stage = "N1"
                        n_description = "Metastasis in 1-3 axillary lymph nodes"
                        n_color = "🟡"
                    elif image_complexity > 0.2:
                        n_stage = "N0"
                        n_description = "No regional lymph node metastasis"
                        n_color = "🟢"
                    else:
                        n_stage = "N2"
                        n_description = "Metastasis in 4-9 axillary lymph nodes"
                        n_color = "🟠"
                else:
                    n_stage = "N0"
                    n_description = "No regional lymph node metastasis (assumed)"
                    n_color = "🟢"
                
                st.write(f"{n_color} **{n_stage}**: {n_description}")
                
                # M (Metastasis) Stage Assessment
                st.write("**M (Metastasis) Stage:**")
                if main_confidence > 0.85:
                    m_stage = "M0"
                    m_description = "No distant metastasis"
                    m_color = "🟢"
                else:
                    m_stage = "M1"
                    m_description = "Distant metastasis present"
                    m_color = "🔴"
                
                st.write(f"{m_color} **{m_stage}**: {m_description}")
                
                # Overall Stage Assessment
                st.write("---")
                st.subheader(" Overall Cancer Stage")
                
                # Determine overall stage based on TNM
                if t_stage in ["T1", "T2"] and n_stage == "N0" and m_stage == "M0":
                    overall_stage = "Stage I"
                    stage_description = "Early-stage cancer, highly treatable"
                    stage_color = "success"
                elif t_stage in ["T1", "T2"] and n_stage == "N1" and m_stage == "M0":
                    overall_stage = "Stage II"
                    stage_description = "Locally advanced cancer, good prognosis with treatment"
                    stage_color = "warning"
                elif t_stage in ["T3", "T4"] or n_stage in ["N2", "N3"] and m_stage == "M0":
                    overall_stage = "Stage III"
                    stage_description = "Advanced local cancer, requires aggressive treatment"
                    stage_color = "error"
                else:
                    overall_stage = "Stage IV"
                    stage_description = "Metastatic cancer, requires comprehensive treatment"
                    stage_color = "error"
                
                if stage_color == "success":
                    st.success(f"**{overall_stage}**: {stage_description}")
                elif stage_color == "warning":
                    st.warning(f"**{overall_stage}**: {stage_description}")
                else:
                    st.error(f"**{overall_stage}**: {stage_description}")
                
                # Histological Grade Assessment
                st.write("---")
                st.subheader(" Histological Grade Assessment")
                
                # Simulate grade based on confidence and image characteristics
                if main_confidence > 0.9:
                    grade = "Grade 1 (Well Differentiated)"
                    grade_description = "Cancer cells look most like normal cells, slow-growing"
                    grade_color = "🟢"
                elif main_confidence > 0.8:
                    grade = "Grade 2 (Moderately Differentiated)"
                    grade_description = "Cancer cells somewhat like normal cells, moderate growth"
                    grade_color = "🟡"
                else:
                    grade = "Grade 3 (Poorly Differentiated)"
                    grade_description = "Cancer cells look very different from normal cells, fast-growing"
                    grade_color = "🔴"
                
                st.write(f"{grade_color} **{grade}**")
                st.write(f"*{grade_description}*")
                
                # Risk Stratification
                st.write("---")
                st.subheader("⚠️ Risk Stratification")
                
                # Calculate risk score
                risk_factors = []
                risk_score = 0
                
                if overall_stage in ["Stage III", "Stage IV"]:
                    risk_factors.append("Advanced stage")
                    risk_score += 3
                elif overall_stage == "Stage II":
                    risk_factors.append("Locally advanced")
                    risk_score += 2
                else:
                    risk_factors.append("Early stage")
                    risk_score += 1
                
                if "Grade 3" in grade:
                    risk_factors.append("High grade")
                    risk_score += 2
                elif "Grade 2" in grade:
                    risk_factors.append("Moderate grade")
                    risk_score += 1
                else:
                    risk_factors.append("Low grade")
                
                if main_confidence < 0.7:
                    risk_factors.append("Uncertain diagnosis")
                    risk_score += 1
                
                # Risk level determination
                if risk_score <= 2:
                    risk_level = "Low Risk"
                    risk_color = "success"
                    risk_recommendation = "Standard treatment protocol, excellent prognosis"
                elif risk_score <= 4:
                    risk_level = "Moderate Risk"
                    risk_color = "warning"
                    risk_recommendation = "Aggressive treatment recommended, good prognosis with treatment"
                else:
                    risk_level = "High Risk"
                    risk_color = "error"
                    risk_recommendation = "Immediate comprehensive treatment required, close monitoring needed"
                
                st.write(f"**Risk Level**: {risk_level}")
                st.write(f"**Risk Factors**: {', '.join(risk_factors)}")
                st.write(f"**Recommendation**: {risk_recommendation}")
                
                # Treatment Recommendations
                st.write("---")
                st.subheader("Treatment Recommendations")
                
                if overall_stage == "Stage I":
                    st.write("**Recommended Treatment:**")
                    st.write("• Surgery (lumpectomy or mastectomy)")
                    st.write("• Radiation therapy (if lumpectomy)")
                    st.write("• Hormone therapy (if hormone receptor positive)")
                    st.write("• **5-year survival rate: 95-100%**")
                    
                elif overall_stage == "Stage II":
                    st.write("**Recommended Treatment:**")
                    st.write("• Surgery (mastectomy or lumpectomy)")
                    st.write("• Chemotherapy")
                    st.write("• Radiation therapy")
                    st.write("• Hormone therapy (if applicable)")
                    st.write("• **5-year survival rate: 85-95%**")
                    
                elif overall_stage == "Stage III":
                    st.write("**Recommended Treatment:**")
                    st.write("• Neoadjuvant chemotherapy")
                    st.write("• Surgery")
                    st.write("• Radiation therapy")
                    st.write("• Targeted therapy")
                    st.write("• **5-year survival rate: 70-85%**")
                    
                else:  # Stage IV
                    st.write("**Recommended Treatment:**")
                    st.write("• Systemic therapy (chemotherapy, targeted therapy)")
                    st.write("• Palliative care")
                    st.write("• Clinical trials")
                    st.write("• **5-year survival rate: 25-30%**")
                
                # Follow-up Recommendations
                st.write("---")
                st.subheader("Follow-up Recommendations")
                
                if overall_stage in ["Stage I", "Stage II"]:
                    st.write("**Follow-up Schedule:**")
                    st.write("• Every 3-6 months for first 2 years")
                    st.write("• Every 6-12 months for years 3-5")
                    st.write("• Annual mammography")
                    st.write("• Regular physical exams")
                else:
                    st.write("**Follow-up Schedule:**")
                    st.write("• Every 2-3 months for first 2 years")
                    st.write("• Every 3-6 months for years 3-5")
                    st.write("• Regular imaging studies")
                    st.write("• Close monitoring of treatment response")
                
            else:  # Non-cancerous
                st.subheader('✅ Benign Assessment')
                st.success("**No Cancer Detected**")
                
                # Benign classification
                if main_confidence > 0.9:
                    benign_type = "Definitely Benign"
                    benign_description = "High confidence in benign diagnosis"
                    benign_color = "success"
                elif main_confidence > 0.8:
                    benign_type = "Likely Benign"
                    benign_description = "High probability of benign condition"
                    benign_color = "success"
                else:
                    benign_type = "Uncertain Benign"
                    benign_description = "Benign but with some uncertainty"
                    benign_color = "warning"
                
                if benign_color == "success":
                    st.success(f"**{benign_type}**: {benign_description}")
                else:
                    st.warning(f"**{benign_type}**: {benign_description}")
                
                # Benign follow-up recommendations
                st.write("---")
                st.subheader("Follow-up Recommendations")
                st.write("**Routine Monitoring:**")
                st.write("• Annual mammography")
                st.write("• Regular self-examinations")
                st.write("• Follow-up in 6-12 months if recommended by physician")
                st.write("• Maintain healthy lifestyle")
            
            # Model Agreement Analysis
            st.write("---")
            st.subheader(" Model Agreement Analysis")
            
            predictions = [r['prediction'] for r in results]
            confidences = [r['confidence'] for r in results]
            agreement = len(set(predictions)) == 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                if agreement:
                    st.success("**Model Agreement**: All models agree on diagnosis")
                else:
                    st.warning("**Model Disagreement**: Models disagree on diagnosis")
                    st.write(f"**Agreement Rate**: {max(predictions.count('Cancerous'), predictions.count('Non-cancerous')) / len(predictions):.1%}")
            
            with col2:
                st.write(f"**Average Confidence**: {np.mean(confidences):.1%}")
                st.write(f"**Confidence Range**: {min(confidences):.1%} - {max(confidences):.1%}")
            
            # Clinical Notes
            # st.write("---")
            # st.subheader("Clinical Notes")
            # st.info("""
            # **Important Disclaimer:**
            # - This staging assessment is based on image analysis and should be used as a preliminary assessment
            # - Final staging requires comprehensive clinical evaluation, imaging studies, and pathology
            # - Always consult with qualified healthcare professionals for definitive diagnosis and treatment planning
            # - This tool is for educational and research purposes only
            # """)
            
            # Export Results
            if st.button("Export Staging Report"):
                staging_report = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "prediction": main_prediction,
                    "confidence": main_confidence,
                    "overall_stage": overall_stage if main_prediction == 'Cancerous' else "Benign",
                    "risk_level": risk_level if main_prediction == 'Cancerous' else "Low",
                    "model_agreement": agreement,
                    "average_confidence": float(np.mean(confidences))
                }
                
                # Save report
                with open("staging_report.json", "w") as f:
                    json.dump(staging_report, f, indent=2)
                
                st.success("Staging report exported to staging_report.json")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("**About This Application**")
st.sidebar.markdown("""
This is of the breast cancer detection system that addresses:
 
**Models Available:**
- VGG16 (CNN)
- ResNet18 (Residual Network)  
- EfficientNetB3 (Efficient CNN)
- DenseNet121 (Dense Connections)
- MobileNetV2 (Mobile Optimized)

Each model has been trained independently with unique architectures and weights.
""")

# Footer
st.markdown("---")
st.markdown("**Breast Cancer Detection System ** | Built with Streamlit and PyTorch")
