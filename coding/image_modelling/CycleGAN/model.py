from torch import nn
import torch

class ResidualBlock(nn.Module):
    """Residual block with instance normalization for CycleGAN generators."""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    ResNet-based generator for CycleGAN.
    Input: (B, 3, 256, 256) in [-1, 1]
    Output: (B, 3, 256, 256) in [-1, 1]
    """
    def __init__(self, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_channels = 64
        for _ in range(2):
            out_channels = in_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]
        
        # Upsampling
        for _ in range(2):
            out_channels = in_channels // 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()  # Output in [-1, 1]
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    PatchGAN discriminator for CycleGAN.
    Input: (B, 3, 256, 256) in [-1, 1]
    Output: (B, 1, 16, 16) - patch of predictions (70x70 receptive field per patch)
    Each pixel in the output represents discriminator confidence for a 70x70 patch of the input.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),  # Pad to get correct output size
            nn.Conv2d(512, 1, 4, padding=1)  # Output: (B, 1, 16, 16) patch
        )

    def forward(self, x):
        return self.model(x)  # Output: (B, 1, 16, 16) - patch of predictions

# Create the 4 models for CycleGAN
def create_models():
    """
    Creates all 4 models needed for CycleGAN:
    - G_A: Generator that converts horses -> zebras
    - G_B: Generator that converts zebras -> horses
    - D_A: Discriminator for horses (real vs fake)
    - D_B: Discriminator for zebras (real vs fake)
    """
    G_A = Generator(n_residual_blocks=9)  # Horse -> Zebra
    G_B = Generator(n_residual_blocks=9)  # Zebra -> Horse
    D_A = Discriminator()                  # Discriminator for horses
    D_B = Discriminator()                  # Discriminator for zebras
    return G_A, G_B, D_A, D_B
