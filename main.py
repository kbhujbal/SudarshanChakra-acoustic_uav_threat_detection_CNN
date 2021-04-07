#!/usr/bin/env python3
"""
SUDARSHANchakra - Acoustic UAV Threat Detection System
======================================================

Main entry point for the defense-grade acoustic surveillance system.

Named after the divine discus weapon of Lord Vishnu - the Sudarshana Chakra -
this system serves as a protective guardian, detecting aerial threats
through acoustic signature analysis.

Usage:
------
    # Complete pipeline: ingest data, train model, evaluate
    python main.py --train

    # Data ingestion only
    python main.py --ingest

    # Inference on audio file
    python main.py --detect path/to/audio.wav

    # Run demonstration
    python main.py --demo

Author: Defense AI Systems
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config


def run_data_ingestion():
    """Execute data ingestion pipeline."""
    from src.data_ingestion import run_ingestion
    return run_ingestion()


def run_training():
    """Execute complete training pipeline."""
    from src.train import run_training as train_pipeline
    return train_pipeline()


def run_inference(audio_path: str, threshold: float = None):
    """Run inference on audio file."""
    from src.inference import analyze_file
    threshold = threshold or Config.THREAT_CONFIDENCE_THRESHOLD
    return analyze_file(audio_path, threshold=threshold)


def run_demo():
    """Run demonstration."""
    from src.inference import demo_inference
    return demo_inference()


def print_banner():
    """Print system banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███████╗██╗   ██╗██████╗  █████╗ ██████╗ ███████╗██╗  ██╗  ║
    ║   ██╔════╝██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝██║  ██║  ║
    ║   ███████╗██║   ██║██║  ██║███████║██████╔╝███████╗███████║  ║
    ║   ╚════██║██║   ██║██║  ██║██╔══██║██╔══██╗╚════██║██╔══██║  ║
    ║   ███████║╚██████╔╝██████╔╝██║  ██║██║  ██║███████║██║  ██║  ║
    ║   ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝  ║
    ║                        CHAKRA                                 ║
    ║                                                              ║
    ║          Acoustic UAV Threat Detection System                ║
    ║              Defense-Grade AI Surveillance                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SUDARSHANchakra - Acoustic UAV Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Commands:
------------------
  --train           Run complete training pipeline (ingest -> train -> evaluate)
  --ingest          Download and prepare dataset only
  --detect FILE     Analyze audio file for threats
  --demo            Run demonstration with sample data

Examples:
---------
  python main.py --train
  python main.py --detect recording.wav
  python main.py --detect recording.wav --threshold 0.3
  python main.py --demo

Configuration:
--------------
  Edit configs/config.py to modify system parameters.
        """,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run complete training pipeline",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run data ingestion only",
    )
    parser.add_argument(
        "--detect",
        type=str,
        metavar="FILE",
        help="Path to audio file for threat detection",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Detection threshold (default: {Config.THREAT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Display current configuration",
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Execute requested operation
    if args.config:
        Config.print_config()

    elif args.ingest:
        print("\n[MODE] Data Ingestion")
        run_data_ingestion()

    elif args.train:
        print("\n[MODE] Training Pipeline")
        print("\nStep 1: Data Ingestion")
        ingestion = run_data_ingestion()

        if ingestion and ingestion.validate_dataset():
            print("\nStep 2: Model Training")
            run_training()
        else:
            print("\n[ERROR] Dataset validation failed. Cannot proceed with training.")
            sys.exit(1)

    elif args.detect:
        print("\n[MODE] Threat Detection")
        run_inference(args.detect, args.threshold)

    elif args.demo:
        print("\n[MODE] Demonstration")
        run_demo()

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  1. Train the model:  python main.py --train")
        print("  2. Run detection:    python main.py --detect audio.wav")
        print("=" * 60)


if __name__ == "__main__":
    main()
