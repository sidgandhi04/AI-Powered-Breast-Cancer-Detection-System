#!/usr/bin/env python3
"""
GAN Training Script for Breast Cancer Detection
This script trains a DCGAN to generate synthetic breast cancer histopathology images
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from gan_models import BreastCancerGAN, BreastCancerDataset, get_transforms

def create_sample_data():
    """Create sample data directory structure if it doesn't exist"""
    data_dir = "data/breast_cancer"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories for different classes
    os.makedirs(os.path.join(data_dir, "benign"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "malignant"), exist_ok=True)
    
    print(f"Created data directory structure at {data_dir}")
    print("Please add your breast cancer histopathology images to the appropriate subdirectories:")
    print(f"- {os.path.join(data_dir, 'benign')} for non-cancerous images")
    print(f"- {os.path.join(data_dir, 'malignant')} for cancerous images")

def train_gan_model(data_dir, epochs=100, batch_size=32, img_size=224, latent_dim=100, 
                   learning_rate=0.0002, beta1=0.5, beta2=0.999, save_interval=10):
    """
    Train the GAN model
    
    Args:
        data_dir: Path to the dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size (assumes square images)
        latent_dim: Dimension of the latent noise vector
        learning_rate: Learning rate for optimizers
        beta1, beta2: Beta parameters for Adam optimizer
        save_interval: Save model every N epochs
    """
    
    print("=" * 60)
    print("GAN Training for Breast Cancer Detection")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        create_sample_data()
        return
    
    # Check for images in the directory
    image_count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
    
    if image_count == 0:
        print(f"No images found in {data_dir}!")
        print("Please add breast cancer histopathology images to the directory.")
        return
    
    print(f"Found {image_count} images in {data_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    transform = get_transforms(img_size)
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = BreastCancerDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    # Initialize GAN
    print("Initializing GAN...")
    gan = BreastCancerGAN(
        latent_dim=latent_dim, 
        img_size=img_size, 
        device=device
    )
    
    # Update learning rates
    for param_group in gan.optimizer_G.param_groups:
        param_group['lr'] = learning_rate
    for param_group in gan.optimizer_D.param_groups:
        param_group['lr'] = learning_rate
    
    print(f"Generator parameters: {sum(p.numel() for p in gan.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('generated_samples', exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Generate fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = gan.generator(noise)
            
            # Train discriminator
            d_loss = gan.train_discriminator(real_imgs, fake_imgs)
            
            # Train generator
            g_loss = gan.train_generator(batch_size)
            
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            num_batches += 1
            
            # Print progress
            if i % max(1, len(dataloader) // 10) == 0:
                print(f'Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(dataloader)}] '
                      f'D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}')
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        gan.g_losses.append(avg_g_loss)
        gan.d_losses.append(avg_d_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}')
        
        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            checkpoint_name = f'gan_epoch_{epoch+1}.pth'
            gan.save_models(checkpoint_name)
            
            # Generate and save samples
            sample_path = f'generated_samples/samples_epoch_{epoch+1}.png'
            gan.generate_samples(epoch + 1, num_samples=16, save_path=sample_path)
            print(f"Generated samples saved to {sample_path}")
    
    # Save final model
    print("\nSaving final model...")
    gan.save_models('gan_breast_cancer_final.pth')
    
    # Plot training history
    print("Plotting training history...")
    gan.plot_training_history()
    
    # Generate final samples
    print("Generating final samples...")
    gan.generate_samples(epochs, num_samples=25, save_path='generated_samples/final_samples.png')
    
    print("\nTraining completed successfully!")
    print(f"Final model saved to: models/gan_breast_cancer_final.pth")
    print(f"Training history plot saved to: gan_training_history.png")
    print(f"Final samples saved to: generated_samples/final_samples.png")

def main():
    parser = argparse.ArgumentParser(description='Train GAN for Breast Cancer Detection')
    parser.add_argument('--data_dir', type=str, default='data/breast_cancer',
                       help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (assumes square images)')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Dimension of the latent noise vector')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 parameter for Adam optimizer')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Beta1: {args.beta1}")
    print(f"  Beta2: {args.beta2}")
    print(f"  Save interval: {args.save_interval}")
    
    # Start training
    train_gan_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()
