"""
SudarshanChakra - Configuration Module
Defense-Grade Acoustic UAV Threat Detection System

All system parameters centralized for operational deployment.
"""

import os
from pathlib import Path


class Config:
    """
    Central configuration for the acoustic threat detection system.

    Parameters are tuned for optimal drone signature detection
    in the 4-8kHz frequency range typical of UAV propellers.
    """

    # ===========================================
    # PROJECT PATHS
    # ===========================================
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODEL_DIR = OUTPUT_DIR / "models"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    LOGS_DIR = OUTPUT_DIR / "logs"

    # Dataset Configuration
    DATASET_REPO_URL = "https://github.com/saraalemadi/DroneAudioDataset.git"
    DATASET_NAME = "DroneAudioDataset"

    # ===========================================
    # AUDIO PROCESSING PARAMETERS
    # ===========================================
    SAMPLE_RATE = 22050          # Hz - Standard for audio ML
    DURATION = 2.0               # Seconds - Analysis window
    N_SAMPLES = int(SAMPLE_RATE * DURATION)  # Total samples per clip

    # Mel-Spectrogram Configuration
    N_FFT = 2048                  # FFT window size
    HOP_LENGTH = 512              # Stride between FFT windows
    N_MELS = 128                  # Number of Mel frequency bins
    F_MIN = 20                    # Minimum frequency (Hz)
    F_MAX = 8000                  # Maximum frequency (Hz) - Captures drone signatures

    # ===========================================
    # MODEL ARCHITECTURE
    # ===========================================
    MODEL_TYPE = "custom_cnn"     # Options: "custom_cnn", "resnet18"
    INPUT_CHANNELS = 1            # Mono audio -> single channel spectrogram
    NUM_CLASSES = 2               # Binary: Drone (1) vs Safe (0)
    DROPOUT_RATE = 0.5            # Regularization

    # ===========================================
    # TRAINING PARAMETERS
    # ===========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5           # L2 regularization

    # Early Stopping Configuration
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Data Split Ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ===========================================
    # DEFENSE-GRADE THRESHOLDS
    # ===========================================
    # In defense applications, missing a threat (False Negative) is catastrophic
    # We bias toward higher recall at the cost of some precision
    THREAT_CONFIDENCE_THRESHOLD = 0.4  # Lower threshold = fewer missed threats

    # Class Labels
    CLASS_NAMES = ["Safe", "Threat"]
    SAFE_LABEL = 0
    THREAT_LABEL = 1

    # ===========================================
    # SYSTEM CONFIGURATION
    # ===========================================
    RANDOM_SEED = 42
    NUM_WORKERS = 4               # DataLoader workers
    PIN_MEMORY = True             # GPU memory optimization

    @classmethod
    def create_directories(cls):
        """Create all necessary project directories."""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.PLOTS_DIR,
            cls.LOGS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_device(cls):
        """Get the best available compute device."""
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        return torch.device("cpu")

    @classmethod
    def print_config(cls):
        """Display current configuration."""
        print("\n" + "=" * 60)
        print("SUDARSHANchakra - SYSTEM CONFIGURATION")
        print("=" * 60)
        print(f"Sample Rate:      {cls.SAMPLE_RATE} Hz")
        print(f"Duration:         {cls.DURATION} s")
        print(f"Mel Bins:         {cls.N_MELS}")
        print(f"Frequency Range:  {cls.F_MIN}-{cls.F_MAX} Hz")
        print(f"Batch Size:       {cls.BATCH_SIZE}")
        print(f"Learning Rate:    {cls.LEARNING_RATE}")
        print(f"Device:           {cls.get_device()}")
        print("=" * 60 + "\n")
