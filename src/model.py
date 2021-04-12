"""
SudarshanChakra - Neural Network Model Module
CNN architectures for acoustic UAV threat detection.

Contains:
- Custom 4-layer CNN optimized for Mel-Spectrogram analysis
- Lightweight ResNet18 variant for transfer learning
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm and ReLU activation.

    Architecture: Conv2D -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # No bias when using BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x


class DroneDetectorCNN(nn.Module):
    """
    Custom 4-layer CNN for acoustic drone detection.

    Designed specifically for Mel-Spectrogram inputs with:
    - Progressive channel expansion (32 -> 64 -> 128 -> 256)
    - Spatial pyramid pooling for variable input sizes
    - Dropout regularization for robust deployment

    Input: Mel-Spectrogram tensor of shape (batch, 1, n_mels, time_frames)
    Output: Binary classification logits (batch, 2)
    """

    def __init__(
        self,
        input_channels: int = Config.INPUT_CHANNELS,
        num_classes: int = Config.NUM_CLASSES,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        # Convolutional feature extractor
        # Progressive channel expansion captures increasingly complex patterns
        self.conv1 = ConvBlock(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, padding=1)

        # Global Average Pooling - handles variable spectrogram lengths
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames)

        Returns:
            Classification logits of shape (batch, num_classes)
        """
        # Feature extraction
        x = self.conv1(x)  # -> (batch, 32, n_mels/2, time/2)
        x = self.conv2(x)  # -> (batch, 64, n_mels/4, time/4)
        x = self.conv3(x)  # -> (batch, 128, n_mels/8, time/8)
        x = self.conv4(x)  # -> (batch, 256, n_mels/16, time/16)

        # Global pooling
        x = self.global_pool(x)  # -> (batch, 256, 1, 1)

        # Classification
        x = self.classifier(x)  # -> (batch, num_classes)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (useful for analysis/visualization).

        Args:
            x: Input tensor

        Returns:
            Feature tensor of shape (batch, 256)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper networks.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with projection if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class LightResNet(nn.Module):
    """
    Lightweight ResNet variant for acoustic analysis.

    Smaller than standard ResNet18 but maintains residual connections
    for stable gradient flow during training.
    """

    def __init__(
        self,
        input_channels: int = Config.INPUT_CHANNELS,
        num_classes: int = Config.NUM_CLASSES,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a layer of residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x


def get_model(model_type: str = Config.MODEL_TYPE) -> nn.Module:
    """
    Factory function to create model instances.

    Args:
        model_type: "custom_cnn" or "resnet18"

    Returns:
        Initialized model
    """
    models = {
        "custom_cnn": DroneDetectorCNN,
        "resnet18": LightResNet,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    model = models[model_type]()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[MODEL] {model_type}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def test_model():
    """Test model with synthetic input."""
    print("\n[TEST] Model architecture test...")

    # Create sample input (batch_size, channels, n_mels, time_frames)
    # Time frames ≈ (sample_rate * duration) / hop_length
    time_frames = int((Config.SAMPLE_RATE * Config.DURATION) / Config.HOP_LENGTH) + 1
    sample_input = torch.randn(4, 1, Config.N_MELS, time_frames)

    print(f"  Input shape: {sample_input.shape}")

    # Test Custom CNN
    print("\n  Testing DroneDetectorCNN...")
    cnn_model = DroneDetectorCNN()
    cnn_output = cnn_model(sample_input)
    print(f"  Output shape: {cnn_output.shape}")

    # Test LightResNet
    print("\n  Testing LightResNet...")
    resnet_model = LightResNet()
    resnet_output = resnet_model(sample_input)
    print(f"  Output shape: {resnet_output.shape}")

    print("\n[SUCCESS] Model tests passed!")


if __name__ == "__main__":
    test_model()
