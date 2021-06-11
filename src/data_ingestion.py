"""
SudarshanChakra - Data Ingestion Module
Automatic dataset acquisition and validation for DroneAudioDataset.

This module handles:
1. Automatic cloning of the DroneAudioDataset repository
2. Directory structure discovery and validation
3. Locating Binary_Drone_Audio data paths dynamically
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config


class DataIngestion:
    """
    Handles automatic dataset acquisition and path discovery.

    The DroneAudioDataset by Sara Al-Emadi contains:
    - Binary classification data (Drone vs Background/Unknown)
    - Multi-class data for different drone types

    This class focuses on the Binary_Drone_Audio subset.
    """

    def __init__(self):
        self.data_dir = Config.DATA_DIR
        self.repo_url = Config.DATASET_REPO_URL
        self.dataset_name = Config.DATASET_NAME
        self.dataset_path = self.data_dir / self.dataset_name

        # Potential folder names for binary classification
        self.drone_keywords = ["drone", "uav", "quadcopter"]
        self.background_keywords = ["background", "unknown", "noise", "ambient", "safe"]

    def clone_repository(self) -> bool:
        """
        Clone the DroneAudioDataset repository if not present.

        Returns:
            bool: True if successful or already exists
        """
        Config.create_directories()

        if self.dataset_path.exists():
            print(f"[INFO] Dataset already exists at: {self.dataset_path}")
            return True

        print(f"[INFO] Cloning DroneAudioDataset from GitHub...")
        print(f"[INFO] Repository: {self.repo_url}")
        print(f"[INFO] Destination: {self.dataset_path}")

        try:
            result = subprocess.run(
                ["git", "clone", self.repo_url, str(self.dataset_path)],
                capture_output=True,
                text=True,
                check=True
            )
            print("[SUCCESS] Repository cloned successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Git clone failed: {e.stderr}")
            return False

        except FileNotFoundError:
            print("[ERROR] Git is not installed. Please install Git and retry.")
            return False

    def print_directory_tree(self, max_depth: int = 4) -> None:
        """
        Print the directory structure of the cloned dataset.

        Args:
            max_depth: Maximum depth to traverse
        """
        print("\n" + "=" * 60)
        print("DATASET DIRECTORY STRUCTURE")
        print("=" * 60)

        if not self.dataset_path.exists():
            print("[ERROR] Dataset not found. Run clone_repository() first.")
            return

        for root, dirs, files in os.walk(self.dataset_path):
            # Calculate depth
            depth = root.replace(str(self.dataset_path), "").count(os.sep)
            if depth > max_depth:
                continue

            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            indent = "│   " * depth
            folder_name = os.path.basename(root)
            print(f"{indent}├── {folder_name}/")

            # Show files (limit to prevent overflow)
            sub_indent = "│   " * (depth + 1)
            wav_files = [f for f in files if f.endswith(".wav")]
            other_files = [f for f in files if not f.endswith(".wav")]

            for f in other_files[:5]:
                print(f"{sub_indent}├── {f}")

            if wav_files:
                print(f"{sub_indent}├── [{len(wav_files)} .wav files]")

        print("=" * 60 + "\n")

    def find_audio_directories(self) -> Dict[str, List[Path]]:
        """
        Dynamically discover directories containing audio files.

        Returns:
            Dict mapping category to list of directories
        """
        audio_dirs = {"drone": [], "background": [], "unknown": []}

        if not self.dataset_path.exists():
            print("[ERROR] Dataset not found.")
            return audio_dirs

        for root, dirs, files in os.walk(self.dataset_path):
            # Check if directory contains wav files
            wav_files = [f for f in files if f.lower().endswith(".wav")]
            if not wav_files:
                continue

            root_path = Path(root)
            dir_name = root_path.name.lower()
            parent_name = root_path.parent.name.lower() if root_path.parent else ""

            # Categorize based on folder naming
            is_drone = any(kw in dir_name for kw in self.drone_keywords)
            is_background = any(kw in dir_name for kw in self.background_keywords)

            # Also check parent directory for context (e.g., Binary_Drone_Audio/Drone)
            if not is_drone and not is_background:
                is_drone = any(kw in parent_name for kw in self.drone_keywords)
                is_background = any(kw in parent_name for kw in self.background_keywords)

            if is_drone:
                audio_dirs["drone"].append(root_path)
            elif is_background:
                audio_dirs["background"].append(root_path)
            else:
                audio_dirs["unknown"].append(root_path)

        return audio_dirs

    def find_binary_data_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Locate the specific paths for binary classification data.

        Searches for directories containing 'Drone' and 'Background'/'Unknown'
        audio files, prioritizing the Binary_Drone_Audio folder structure.

        Returns:
            Tuple of (drone_path, background_path) or (None, None) if not found
        """
        print("\n[INFO] Searching for binary classification audio data...")

        drone_path = None
        background_path = None

        # First, look for Binary_Drone_Audio specific structure
        for root, dirs, files in os.walk(self.dataset_path):
            root_path = Path(root)
            root_name_lower = root_path.name.lower()

            # Check if this directory contains wav files
            wav_files = [f for f in files if f.lower().endswith(".wav")]

            if wav_files:
                # Priority 1: Exact match for "Drone" folder in Binary path
                if "binary" in str(root_path).lower():
                    if root_name_lower == "drone" or "drone" in root_name_lower:
                        if not any(kw in root_name_lower for kw in self.background_keywords):
                            drone_path = root_path
                            print(f"[FOUND] Drone audio: {root_path} ({len(wav_files)} files)")

                    elif any(kw in root_name_lower for kw in self.background_keywords):
                        background_path = root_path
                        print(f"[FOUND] Background audio: {root_path} ({len(wav_files)} files)")

        # Fallback: Search more broadly if binary-specific paths not found
        if not drone_path or not background_path:
            print("[INFO] Performing broader search...")

            for root, dirs, files in os.walk(self.dataset_path):
                root_path = Path(root)
                root_name_lower = root_path.name.lower()
                wav_files = [f for f in files if f.lower().endswith(".wav")]

                if wav_files:
                    # Look for drone directories
                    if not drone_path:
                        if root_name_lower == "drone" or (
                            "drone" in root_name_lower and
                            not any(kw in root_name_lower for kw in self.background_keywords)
                        ):
                            drone_path = root_path
                            print(f"[FOUND] Drone audio (fallback): {root_path}")

                    # Look for background directories
                    if not background_path:
                        if any(kw == root_name_lower for kw in self.background_keywords) or \
                           any(kw in root_name_lower for kw in self.background_keywords):
                            background_path = root_path
                            print(f"[FOUND] Background audio (fallback): {root_path}")

        # Final validation
        if drone_path and background_path:
            print("\n[SUCCESS] Binary classification data located!")
            print(f"  Drone Path:      {drone_path}")
            print(f"  Background Path: {background_path}")
        else:
            print("\n[WARNING] Could not locate both required directories.")
            print("  Please verify the dataset structure manually.")

        return drone_path, background_path

    def get_all_audio_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Get lists of all drone and background audio files.

        Returns:
            Tuple of (drone_files, background_files)
        """
        drone_path, background_path = self.find_binary_data_paths()

        drone_files = []
        background_files = []

        if drone_path and drone_path.exists():
            drone_files = list(drone_path.glob("*.wav")) + list(drone_path.glob("*.WAV"))

        if background_path and background_path.exists():
            background_files = list(background_path.glob("*.wav")) + list(background_path.glob("*.WAV"))

        print(f"\n[DATA SUMMARY]")
        print(f"  Drone samples:      {len(drone_files)}")
        print(f"  Background samples: {len(background_files)}")
        print(f"  Total samples:      {len(drone_files) + len(background_files)}")

        return drone_files, background_files

    def validate_dataset(self) -> bool:
        """
        Validate that the dataset is properly downloaded and structured.

        Returns:
            bool: True if dataset is valid
        """
        print("\n[INFO] Validating dataset integrity...")

        drone_files, background_files = self.get_all_audio_files()

        if len(drone_files) == 0:
            print("[ERROR] No drone audio files found!")
            return False

        if len(background_files) == 0:
            print("[ERROR] No background audio files found!")
            return False

        # Check file accessibility
        sample_drone = drone_files[0]
        sample_background = background_files[0]

        try:
            # Verify files are readable
            with open(sample_drone, "rb") as f:
                header = f.read(4)
                if header != b"RIFF":
                    print(f"[WARNING] {sample_drone} may not be a valid WAV file")

            with open(sample_background, "rb") as f:
                header = f.read(4)
                if header != b"RIFF":
                    print(f"[WARNING] {sample_background} may not be a valid WAV file")

            print("[SUCCESS] Dataset validation passed!")
            return True

        except Exception as e:
            print(f"[ERROR] File validation failed: {e}")
            return False


def run_ingestion():
    """Execute the complete data ingestion pipeline."""
    print("\n" + "=" * 60)
    print("SUDARSHANchakra - DATA INGESTION PIPELINE")
    print("Acoustic UAV Threat Detection System")
    print("=" * 60)

    ingestion = DataIngestion()

    # Step 1: Clone repository
    if not ingestion.clone_repository():
        print("[FATAL] Failed to acquire dataset. Exiting.")
        return None

    # Step 2: Print directory structure
    ingestion.print_directory_tree()

    # Step 3: Locate binary data paths
    drone_path, background_path = ingestion.find_binary_data_paths()

    # Step 4: Validate dataset
    if ingestion.validate_dataset():
        print("\n[READY] Dataset is ready for training!")
        return ingestion
    else:
        print("\n[WARNING] Dataset validation failed. Please check the data.")
        return ingestion


if __name__ == "__main__":
    run_ingestion()
