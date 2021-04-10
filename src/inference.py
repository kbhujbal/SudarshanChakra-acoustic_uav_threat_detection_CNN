"""
SudarshanChakra - Inference Module
Real-time acoustic threat detection for production deployment.

Usage:
    python inference.py --audio path/to/audio.wav
    python inference.py --audio path/to/audio.wav --model path/to/model.pth

Features:
- Single file inference with threat assessment
- Batch inference for multiple files
- Confidence scoring with defense-optimized thresholds
- Real-time audio stream processing (placeholder for future)
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Union
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config
from src.model import get_model
from src.data_loader import AudioTransform


class ThreatDetector:
    """
    Production inference engine for UAV acoustic threat detection.

    Optimized for:
    - Low-latency single-file inference
    - Defense-grade confidence thresholding
    - Clear threat/safe status reporting
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        threshold: float = Config.THREAT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the threat detector.

        Args:
            model_path: Path to trained model checkpoint (default: best_model.pth)
            device: Compute device (default: auto-detect)
            threshold: Confidence threshold for threat classification
                       Lower = more sensitive (fewer missed threats)
        """
        self.device = device or Config.get_device()
        self.threshold = threshold
        self.transform = AudioTransform()

        # Load model
        self.model_path = model_path or Config.MODEL_DIR / "best_model.pth"
        self.model = self._load_model()

        print(f"\n[DETECTOR INITIALIZED]")
        print(f"  Model: {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  Threat Threshold: {self.threshold}")

    def _load_model(self) -> torch.nn.Module:
        """Load and prepare model for inference."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using: python -m src.train"
            )

        # Initialize model architecture
        model = get_model(Config.MODEL_TYPE)

        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Set to evaluation mode
        model.eval()
        model.to(self.device)

        return model

    @torch.no_grad()
    def detect(self, audio_path: Union[str, Path]) -> dict:
        """
        Analyze audio file for UAV threat.

        Args:
            audio_path: Path to .wav audio file

        Returns:
            Detection result dictionary:
            {
                "status": "THREAT" or "SAFE",
                "confidence": float (0-1),
                "probabilities": {"safe": float, "threat": float},
                "file": str
            }
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Transform audio to spectrogram
        spectrogram = self.transform(audio_path)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)  # Add batch dim

        # Model inference
        logits = self.model(spectrogram)
        probabilities = F.softmax(logits, dim=1)

        # Extract probabilities
        prob_safe = probabilities[0, Config.SAFE_LABEL].item()
        prob_threat = probabilities[0, Config.THREAT_LABEL].item()

        # Classification with defense-optimized threshold
        # Lower threshold = more sensitive to threats
        is_threat = prob_threat >= self.threshold

        result = {
            "status": "THREAT" if is_threat else "SAFE",
            "confidence": prob_threat if is_threat else prob_safe,
            "probabilities": {
                "safe": round(prob_safe, 4),
                "threat": round(prob_threat, 4),
            },
            "file": str(audio_path.name),
            "threshold_used": self.threshold,
        }

        return result

    def detect_batch(self, audio_paths: List[Path]) -> List[dict]:
        """
        Analyze multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of detection results
        """
        results = []
        for path in audio_paths:
            try:
                result = self.detect(path)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "ERROR",
                    "error": str(e),
                    "file": str(path),
                })
        return results

    def print_alert(self, result: dict) -> None:
        """
        Print formatted alert message.

        Args:
            result: Detection result dictionary
        """
        if result["status"] == "THREAT":
            print("\n" + "!" * 60)
            print("!!! [ALERT] DRONE DETECTED !!!")
            print("!" * 60)
            print(f"  File: {result['file']}")
            print(f"  Threat Confidence: {result['probabilities']['threat']:.1%}")
            print(f"  Status: HOSTILE AERIAL VEHICLE DETECTED")
            print("!" * 60)

        elif result["status"] == "SAFE":
            print("\n" + "-" * 60)
            print("[STATUS] CLEAR - No Threat Detected")
            print("-" * 60)
            print(f"  File: {result['file']}")
            print(f"  Safety Confidence: {result['probabilities']['safe']:.1%}")
            print(f"  Status: Environment appears safe")
            print("-" * 60)

        else:
            print(f"\n[ERROR] Analysis failed for {result['file']}")
            print(f"  Reason: {result.get('error', 'Unknown')}")


def analyze_file(
    audio_path: str,
    model_path: Optional[str] = None,
    threshold: float = Config.THREAT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Convenience function for single-file analysis.

    Args:
        audio_path: Path to audio file
        model_path: Optional path to model checkpoint
        threshold: Detection threshold

    Returns:
        Detection result
    """
    model_path = Path(model_path) if model_path else None
    detector = ThreatDetector(model_path=model_path, threshold=threshold)
    result = detector.detect(audio_path)
    detector.print_alert(result)
    return result


def demo_inference():
    """
    Demonstration of inference capability.

    If no trained model exists, provides instructions.
    If model exists, runs on sample data if available.
    """
    print("\n" + "=" * 60)
    print("SUDARSHANchakra - THREAT DETECTION DEMO")
    print("=" * 60)

    model_path = Config.MODEL_DIR / "best_model.pth"

    if not model_path.exists():
        print("\n[INFO] No trained model found.")
        print("\nTo train the model, run:")
        print("  python main.py --train")
        print("\nOr:")
        print("  python -m src.train")
        return

    # Initialize detector
    try:
        detector = ThreatDetector()
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return

    # Look for sample audio files
    from src.data_ingestion import DataIngestion
    ingestion = DataIngestion()

    if ingestion.dataset_path.exists():
        drone_files, background_files = ingestion.get_all_audio_files()

        print("\n[DEMO] Running inference on sample files...")

        # Test one drone file
        if drone_files:
            print("\n--- Testing Drone Audio ---")
            result = detector.detect(drone_files[0])
            detector.print_alert(result)

        # Test one background file
        if background_files:
            print("\n--- Testing Background Audio ---")
            result = detector.detect(background_files[0])
            detector.print_alert(result)
    else:
        print("\n[INFO] Dataset not found. Run data ingestion first:")
        print("  python -m src.data_ingestion")


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(
        description="SUDARSHANchakra - Acoustic UAV Threat Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze a single file:
    python inference.py --audio recording.wav

  Use custom model and threshold:
    python inference.py --audio recording.wav --model models/v2.pth --threshold 0.3

  Run demo:
    python inference.py --demo
        """,
    )

    parser.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to audio file for analysis",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model checkpoint (default: outputs/models/best_model.pth)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=Config.THREAT_CONFIDENCE_THRESHOLD,
        help=f"Threat confidence threshold (default: {Config.THREAT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration on sample data",
    )

    args = parser.parse_args()

    if args.demo:
        demo_inference()
    elif args.audio:
        analyze_file(args.audio, args.model, args.threshold)
    else:
        parser.print_help()
        print("\n[TIP] Use --demo to test with sample data")


if __name__ == "__main__":
    main()
