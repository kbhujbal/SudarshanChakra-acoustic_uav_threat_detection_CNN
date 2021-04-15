"""
SudarshanChakra - Data Loader Module
Custom PyTorch Dataset for acoustic threat detection.

Converts raw audio waveforms to Mel-Spectrograms on-the-fly
for efficient GPU memory usage during training.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import librosa

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config
from src.data_ingestion import DataIngestion


class AudioTransform:
    """
    Audio preprocessing pipeline for acoustic threat detection.

    Converts raw waveforms to Mel-Spectrograms optimized for
    detecting drone propeller acoustic signatures.
    """

    def __init__(
        self,
        sample_rate: int = Config.SAMPLE_RATE,
        duration: float = Config.DURATION,
        n_mels: int = Config.N_MELS,
        n_fft: int = Config.N_FFT,
        hop_length: int = Config.HOP_LENGTH,
        f_min: float = Config.F_MIN,
        f_max: float = Config.F_MAX,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

    def load_audio(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess audio file.

        Args:
            file_path: Path to .wav file

        Returns:
            Normalized audio array of fixed length
        """
        try:
            # Load audio with target sample rate
            waveform, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True,
                duration=self.duration
            )

            # Pad or truncate to exact length
            if len(waveform) < self.n_samples:
                # Pad with zeros
                padding = self.n_samples - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode="constant")
            elif len(waveform) > self.n_samples:
                # Truncate
                waveform = waveform[:self.n_samples]

            return waveform

        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            # Return silence on error
            return np.zeros(self.n_samples, dtype=np.float32)

    def to_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert waveform to Mel-Spectrogram.

        The Mel scale emphasizes frequencies perceptually important to humans
        and captures the characteristic frequency patterns of drone motors.

        Args:
            waveform: Audio time series

        Returns:
            Log-scaled Mel-Spectrogram as 2D array
        """
        # Compute Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0  # Power spectrogram
        )

        # Convert to log scale (dB)
        # Add small epsilon to avoid log(0)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )

        return mel_spec_norm

    def __call__(self, file_path: Path) -> torch.Tensor:
        """
        Complete transform pipeline: Audio file -> Tensor.

        Args:
            file_path: Path to audio file

        Returns:
            Mel-Spectrogram tensor of shape (1, n_mels, time_frames)
        """
        waveform = self.load_audio(file_path)
        mel_spec = self.to_mel_spectrogram(waveform)

        # Convert to tensor and add channel dimension
        tensor = torch.FloatTensor(mel_spec).unsqueeze(0)

        return tensor


class DroneAudioDataset(Dataset):
    """
    PyTorch Dataset for drone acoustic detection.

    Loads audio files on-demand and converts them to Mel-Spectrograms.
    Supports data augmentation for improved model robustness.
    """

    def __init__(
        self,
        drone_files: List[Path],
        background_files: List[Path],
        transform: Optional[AudioTransform] = None,
        augment: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            drone_files: List of paths to drone audio files
            background_files: List of paths to background audio files
            transform: Audio transform pipeline (default: AudioTransform)
            augment: Whether to apply data augmentation
        """
        self.transform = transform or AudioTransform()
        self.augment = augment

        # Combine files with labels
        # Label 0 = Safe (Background), Label 1 = Threat (Drone)
        self.samples = []

        for file_path in background_files:
            self.samples.append((file_path, Config.SAFE_LABEL))

        for file_path in drone_files:
            self.samples.append((file_path, Config.THREAT_LABEL))

        print(f"[DATASET] Initialized with {len(self.samples)} samples")
        print(f"  - Drone (Threat):     {len(drone_files)}")
        print(f"  - Background (Safe):  {len(background_files)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (mel_spectrogram_tensor, label)
        """
        file_path, label = self.samples[idx]

        # Transform audio to spectrogram
        spectrogram = self.transform(file_path)

        # Apply augmentation if enabled
        if self.augment:
            spectrogram = self._augment(spectrogram)

        return spectrogram, label

    def _augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to spectrogram.

        Augmentations:
        - Time masking: Random time segments set to zero
        - Frequency masking: Random frequency bands set to zero
        - Random noise: Small Gaussian noise addition
        """
        # Time masking (SpecAugment style)
        if torch.rand(1).item() > 0.5:
            t_mask_size = int(spectrogram.shape[2] * 0.1)
            t_start = torch.randint(0, spectrogram.shape[2] - t_mask_size, (1,)).item()
            spectrogram[:, :, t_start:t_start + t_mask_size] = 0

        # Frequency masking
        if torch.rand(1).item() > 0.5:
            f_mask_size = int(spectrogram.shape[1] * 0.1)
            f_start = torch.randint(0, spectrogram.shape[1] - f_mask_size, (1,)).item()
            spectrogram[:, f_start:f_start + f_mask_size, :] = 0

        # Add small noise
        if torch.rand(1).item() > 0.7:
            noise = torch.randn_like(spectrogram) * 0.01
            spectrogram = spectrogram + noise
            spectrogram = torch.clamp(spectrogram, 0, 1)

        return spectrogram

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data handling.

        Returns:
            Tensor of class weights
        """
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels)
        total = len(labels)

        # Inverse frequency weighting
        weights = total / (len(class_counts) * class_counts)

        return torch.FloatTensor(weights)


def get_data_loaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    train_ratio: float = Config.TRAIN_RATIO,
    val_ratio: float = Config.VAL_RATIO,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.

    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Get dataset files
    ingestion = DataIngestion()

    # Ensure data is available
    if not ingestion.dataset_path.exists():
        print("[INFO] Dataset not found. Running ingestion...")
        ingestion.clone_repository()

    drone_files, background_files = ingestion.get_all_audio_files()

    if len(drone_files) == 0 or len(background_files) == 0:
        raise ValueError("Dataset is empty or not properly downloaded!")

    # Shuffle files
    np.random.seed(Config.RANDOM_SEED)
    np.random.shuffle(drone_files)
    np.random.shuffle(background_files)

    # Split each class proportionally
    def split_files(files: List[Path]) -> Tuple[List, List, List]:
        n = len(files)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return files[:train_end], files[train_end:val_end], files[val_end:]

    drone_train, drone_val, drone_test = split_files(drone_files)
    bg_train, bg_val, bg_test = split_files(background_files)

    print(f"\n[DATA SPLIT]")
    print(f"  Training:   {len(drone_train)} drone, {len(bg_train)} background")
    print(f"  Validation: {len(drone_val)} drone, {len(bg_val)} background")
    print(f"  Test:       {len(drone_test)} drone, {len(bg_test)} background")

    # Create datasets
    train_dataset = DroneAudioDataset(drone_train, bg_train, augment=True)
    val_dataset = DroneAudioDataset(drone_val, bg_val, augment=False)
    test_dataset = DroneAudioDataset(drone_test, bg_test, augment=False)

    # Get class weights from training set
    class_weights = train_dataset.get_class_weights()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test the data loader
    print("\n[TEST] Testing DroneAudioDataset...")

    try:
        train_loader, val_loader, test_loader, weights = get_data_loaders(batch_size=4)

        # Get a sample batch
        for batch_x, batch_y in train_loader:
            print(f"\n[SAMPLE BATCH]")
            print(f"  Input shape:  {batch_x.shape}")
            print(f"  Labels:       {batch_y}")
            print(f"  Class weights: {weights}")
            break

        print("\n[SUCCESS] Data loader test passed!")

    except Exception as e:
        print(f"[ERROR] Data loader test failed: {e}")
        raise
