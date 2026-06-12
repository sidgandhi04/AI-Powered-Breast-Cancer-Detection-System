import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Generator(nn.Module):
    """DCGAN Generator for breast cancer image generation"""
    def __init__(self, latent_dim=100, img_channels=3, img_size=224):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Calculate the initial size after upsampling
        # We'll start from 7x7 and upsample to 224x224
        self.init_size = img_size // 32  # 224 // 32 = 7
        
        # Project noise to initial feature map
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        
        # Main generator network
        self.conv_blocks = nn.Sequential(
            # Input: 512 x 7 x 7
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 14 x 14
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 28 x 28
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 56 x 56
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32 x 112 x 112
            
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 224 x 224
        )
        
    def forward(self, noise):
        # Project noise to feature map
        x = self.fc(noise)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(x)
        return img

class Discriminator(nn.Module):
    """DCGAN Discriminator for breast cancer image classification"""
    def __init__(self, img_channels=3, img_size=224):
        super(Discriminator, self).__init__()
        
        def conv_block(in_channels, out_channels, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Main discriminator network
        self.conv_blocks = nn.Sequential(
            # Input: 3 x 224 x 224
            conv_block(img_channels, 32, stride=2, padding=1),
            # 32 x 112 x 112
            
            conv_block(32, 64, stride=2, padding=1),
            # 64 x 56 x 56
            
            conv_block(64, 128, stride=2, padding=1),
            # 128 x 28 x 28
            
            conv_block(128, 256, stride=2, padding=1),
            # 256 x 14 x 14
            
            conv_block(256, 512, stride=2, padding=1),
            # 512 x 7 x 7
        )
        
        # Calculate the size after conv blocks
        self.fc_size = 512 * 7 * 7
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(self.fc_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        x = self.conv_blocks(img)
        x = x.view(x.size(0), -1)
        validity = self.fc(x)
        return validity

class BreastCancerGAN:
    """Main GAN class for training and inference"""
    def __init__(self, latent_dim=100, img_channels=3, img_size=224, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Initialize models
        self.generator = Generator(latent_dim, img_channels, img_size).to(self.device)
        self.discriminator = Discriminator(img_channels, img_size).to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
    def train_discriminator(self, real_imgs, fake_imgs):
        """Train discriminator on real and fake images"""
        self.optimizer_D.zero_grad()
        
        # Real images
        real_labels = torch.ones(real_imgs.size(0), 1).to(self.device)
        real_pred = self.discriminator(real_imgs)
        real_loss = self.criterion(real_pred, real_labels)
        
        # Fake images
        fake_labels = torch.zeros(fake_imgs.size(0), 1).to(self.device)
        fake_pred = self.discriminator(fake_imgs.detach())
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()
    
    def train_generator(self, batch_size):
        """Train generator to fool discriminator"""
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(noise)
        
        # Try to fool discriminator
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_pred = self.discriminator(fake_imgs)
        g_loss = self.criterion(fake_pred, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def train(self, dataloader, epochs=100, save_interval=10):
        """Train the GAN"""
        print(f"Training GAN on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            
            for i, real_imgs in enumerate(dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                
                # Generate fake images
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_imgs = self.generator(noise)
                
                # Train discriminator
                d_loss = self.train_discriminator(real_imgs, fake_imgs)
                
                # Train generator
                g_loss = self.train_generator(batch_size)
                
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1
                
                # Print progress
                if i % 50 == 0:
                    print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] '
                          f'D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}')
            
            # Store losses
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            print(f'Epoch [{epoch}/{epochs}] Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}')
            
            # Save model checkpoints
            if epoch % save_interval == 0:
                self.save_models(f'gan_epoch_{epoch}.pth')
                self.generate_samples(epoch, num_samples=16)
        
        print("Training completed!")
        self.save_models('gan_final.pth')
    
    def generate_samples(self, epoch, num_samples=16, save_path=None):
        """Generate sample images"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_imgs = self.generator(noise)
            
            # Denormalize images
            fake_imgs = (fake_imgs + 1) / 2  # From [-1, 1] to [0, 1]
            fake_imgs = torch.clamp(fake_imgs, 0, 1)
            
            # Create grid - adjust grid size based on num_samples
            grid_size = int(np.ceil(np.sqrt(num_samples)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            
            # Handle different cases for axes
            if grid_size == 1:
                if num_samples == 1:
                    axes = [axes]  # Single subplot
                else:
                    axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Display images
            for i in range(num_samples):
                if i < len(axes):
                    img = fake_imgs[i].cpu().permute(1, 2, 0).numpy()
                    axes[i].imshow(img)
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_samples, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.savefig(f'generated_samples_epoch_{epoch}.png')
            plt.close()
    
    def save_models(self, filename):
        """Save generator and discriminator models"""
        os.makedirs('models', exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer': self.optimizer_G.state_dict(),
            'discriminator_optimizer': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'latent_dim': self.latent_dim,
            'img_size': self.img_size
        }, f'models/{filename}')
        print(f"Models saved to models/{filename}")
    
    def load_models(self, filename):
        """Load generator and discriminator models"""
        # Handle both full paths and just filenames
        if filename.startswith('models/'):
            model_path = filename
        else:
            model_path = f'models/{filename}'
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GAN model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['generator_optimizer'])
            self.optimizer_D.load_state_dict(checkpoint['discriminator_optimizer'])
            self.g_losses = checkpoint.get('g_losses', [])
            self.d_losses = checkpoint.get('d_losses', [])
            print(f"Models loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading GAN models from {model_path}: {str(e)}")
    
    def generate_images(self, num_images=1, num_samples=None, class_label=None):
        """Generate images for data augmentation"""
        # Handle both parameter names for backward compatibility
        if num_samples is not None:
            num_images = num_samples
            
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_dim).to(self.device)
            fake_imgs = self.generator(noise)
            
            # Denormalize images
            fake_imgs = (fake_imgs + 1) / 2  # From [-1, 1] to [0, 1]
            fake_imgs = torch.clamp(fake_imgs, 0, 1)
            
            return fake_imgs.cpu()
    
    def plot_training_history(self):
        """Plot training loss history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('gan_training_history.png')
        plt.show()

class BreastCancerDataset(Dataset):
    """Custom dataset for breast cancer images"""
    def __init__(self, data_dir, transform=None, class_label=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_label = class_label
        self.images = []
        
        # Load images
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(root, file))
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_transforms(img_size=224):
    """Get data transforms for GAN training"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return transform

def train_gan(data_dir, epochs=100, batch_size=32, img_size=224, latent_dim=100):
    """Main training function"""
    # Create dataset and dataloader
    transform = get_transforms(img_size)
    dataset = BreastCancerDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize GAN
    gan = BreastCancerGAN(latent_dim=latent_dim, img_size=img_size)
    
    # Train GAN
    gan.train(dataloader, epochs=epochs)
    
    # Plot training history
    gan.plot_training_history()
    
    return gan

if __name__ == "__main__":
    # Example usage
    data_dir = "data/breast_cancer"  # Update with your data directory
    gan = train_gan(data_dir, epochs=50, batch_size=16)
