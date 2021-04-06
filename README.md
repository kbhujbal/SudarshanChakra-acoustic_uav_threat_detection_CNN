# SUDARSHANchakra - Acoustic UAV Threat Detection System

> Named after the divine discus weapon of Lord Vishnu, this system serves as a protective guardian against aerial threats.

A production-grade deep learning system for detecting drone/UAV acoustic signatures using Mel-Spectrogram analysis and Convolutional Neural Networks.

## Mission

Develop an acoustic surveillance system for ground-based defense units that can:
- Listen to environmental audio in real-time
- Detect the specific acoustic signature of drone propellers
- Classify audio as **Threat (Drone)** or **Safe (Background)**

## Features

- **Audio-to-Vision Pipeline**: Converts raw audio waveforms to Mel-Spectrograms for CNN analysis
- **Custom CNN Architecture**: 4-layer network optimized for acoustic pattern recognition
- **Defense-Grade Metrics**: Prioritizes Recall (minimizing missed threats) over Precision
- **Production Ready**: Modular Python package with inference API
- **Auto-Ingestion**: Automatically downloads and prepares the DroneAudioDataset

## Project Structure

```
SudarshanChakra/
├── configs/
│   └── config.py           # Central configuration parameters
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py   # Auto-clone dataset from GitHub
│   ├── data_loader.py      # PyTorch Dataset & DataLoader
│   ├── model.py            # CNN architectures
│   ├── train.py            # Training pipeline with early stopping
│   └── inference.py        # Real-time threat detection
├── outputs/
│   ├── models/             # Saved model checkpoints
│   ├── plots/              # Confusion matrix, training curves
│   └── logs/               # Training reports (JSON)
├── data/                   # Auto-downloaded dataset
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Complete pipeline: download data -> train -> evaluate
python main.py --train
```

This will:
1. Clone the DroneAudioDataset from GitHub
2. Print the dataset directory structure
3. Auto-discover Drone/Background audio paths
4. Train the CNN with early stopping
5. Generate confusion matrix and training plots
6. Save the best model to `outputs/models/best_model.pth`

### 3. Run Inference

```bash
# Analyze a single audio file
python main.py --detect path/to/audio.wav

# With custom threshold (lower = more sensitive)
python main.py --detect recording.wav --threshold 0.3

# Run demo on sample data
python main.py --demo
```

## Configuration

Edit `configs/config.py` to modify system parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 22050 Hz | Audio sampling rate |
| `DURATION` | 2.0 s | Analysis window |
| `N_MELS` | 128 | Mel frequency bins |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Adam optimizer LR |
| `THREAT_CONFIDENCE_THRESHOLD` | 0.4 | Detection sensitivity |

## Dataset

**Source**: [DroneAudioDataset by Sara Al-Emadi](https://github.com/saraalemadi/DroneAudioDataset)

The system automatically:
1. Clones the repository to `data/DroneAudioDataset/`
2. Scans for `Binary_Drone_Audio` folder structure
3. Discovers Drone and Background/Unknown audio directories
4. Validates WAV file integrity

## Model Architecture

### DroneDetectorCNN (Default)

```
Input: Mel-Spectrogram (1, 128, 87)
    ↓
ConvBlock: 1 → 32 channels, MaxPool
    ↓
ConvBlock: 32 → 64 channels, MaxPool
    ↓
ConvBlock: 64 → 128 channels, MaxPool
    ↓
ConvBlock: 128 → 256 channels, MaxPool
    ↓
Global Average Pooling
    ↓
FC: 256 → 128 → 64 → 2 (with Dropout)
    ↓
Output: [Safe, Threat] logits
```

**Total Parameters**: ~500K (lightweight for edge deployment)

## Defense Metrics

In defense applications:
- **False Negative** (missed drone) = **Critical** → Prioritize high Recall
- **False Positive** (false alarm) = **Acceptable** → Accept lower Precision

The system uses a **0.4 confidence threshold** by default, biasing toward threat detection.

### Output Example

```
[TEST RESULTS]
  Accuracy:  0.9523
  Precision: 0.9412
  Recall:    0.9697  ← 97% of drones detected!
  F1 Score:  0.9552

[DEFENSE METRICS INTERPRETATION]
  - Recall (96.97%): Percentage of actual threats detected
    → 3.0% of threats are MISSED (False Negatives)
  - Precision (94.12%): Percentage of alerts that are real threats
    → 5.9% of alerts are FALSE (False Positives)
```

## Outputs

After training, find these artifacts in `outputs/`:

| File | Description |
|------|-------------|
| `models/best_model.pth` | Best model checkpoint |
| `plots/confusion_matrix.png` | Test set confusion matrix |
| `plots/training_history.png` | Loss, accuracy, metrics curves |
| `logs/training_report.json` | Full training metrics log |

## Inference API

```python
from src.inference import ThreatDetector

# Initialize detector
detector = ThreatDetector(threshold=0.4)

# Analyze audio file
result = detector.detect("recording.wav")

# Result format:
# {
#     "status": "THREAT",  # or "SAFE"
#     "confidence": 0.89,
#     "probabilities": {"safe": 0.11, "threat": 0.89},
#     "file": "recording.wav"
# }

# Print formatted alert
detector.print_alert(result)
```

## CLI Reference

```bash
# Show configuration
python main.py --config

# Data ingestion only
python main.py --ingest

# Training pipeline
python main.py --train

# Inference
python main.py --detect audio.wav
python main.py --detect audio.wav --threshold 0.3

# Demo mode
python main.py --demo
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- librosa 0.10+
- scikit-learn 1.3+
- Git (for dataset cloning)

## Citation

If using the DroneAudioDataset:
```
Al-Emadi, Sara, et al. "Audio Based Drone Detection and Identification
using Deep Learning." 2019 15th International Wireless Communications
& Mobile Computing Conference (IWCMC). IEEE, 2019.
```

## License

This project is for authorized defense research and educational purposes.

---

**SUDARSHANchakra** - Protecting the skies through acoustic intelligence.
