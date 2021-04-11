"""
SudarshanChakra - Training Module
Production-grade training pipeline with defense-optimized evaluation.

Features:
- Early stopping with patience
- Model checkpointing (best validation loss)
- Defense-grade metrics: Precision, Recall, F1
- Confusion matrix visualization
- Training history logging
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config
from src.model import get_model
from src.data_loader import get_data_loaders


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        min_delta: float = Config.EARLY_STOPPING_MIN_DELTA,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Production training pipeline for acoustic threat detection.

    Handles:
    - Training loop with progress tracking
    - Validation and evaluation
    - Model checkpointing
    - Metrics logging and visualization
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or Config.get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss function with class weights for imbalanced data
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Early stopping
        self.early_stopping = EarlyStopping()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "learning_rate": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_model_path = Config.MODEL_DIR / "best_model.pth"

        # Create output directories
        Config.create_directories()

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            loader: DataLoader for validation/test set

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        avg_loss = total_loss / len(loader.dataset)
        accuracy = (all_predictions == all_labels).mean()

        # Defense-critical metrics
        # In defense: Recall is critical (don't miss threats!)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_predictions,
            "labels": all_labels,
        }

    def train(self, num_epochs: int = Config.NUM_EPOCHS) -> Dict:
        """
        Complete training loop.

        Args:
            num_epochs: Maximum number of epochs

        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("SUDARSHANchakra - TRAINING INITIATED")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
        print("=" * 60 + "\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_metrics = self.validate(self.val_loader)

            # Update learning rate
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["learning_rate"].append(current_lr)

            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Precision:  {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            print(f"  F1 Score:   {val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.2e}")

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(self.best_model_path, epoch, val_metrics)
                print(f"  [CHECKPOINT] Best model saved!")

            # Early stopping check
            if self.early_stopping(val_metrics["loss"]):
                print(f"\n[EARLY STOPPING] No improvement for {Config.EARLY_STOPPING_PATIENCE} epochs.")
                break

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": {
                "sample_rate": Config.SAMPLE_RATE,
                "n_mels": Config.N_MELS,
                "duration": Config.DURATION,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def evaluate(self) -> Dict[str, float]:
        """
        Final evaluation on test set.

        Returns:
            Test metrics dictionary
        """
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best model
        if self.best_model_path.exists():
            print(f"Loading best model from: {self.best_model_path}")
            self.load_checkpoint(self.best_model_path)

        test_metrics = self.validate(self.test_loader)

        print(f"\n[TEST RESULTS]")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")

        # Defense context explanation
        print("\n[DEFENSE METRICS INTERPRETATION]")
        print(f"  - Recall ({test_metrics['recall']:.2%}): Percentage of actual threats detected")
        print(f"    -> {(1 - test_metrics['recall']) * 100:.1f}% of threats are MISSED (False Negatives)")
        print(f"  - Precision ({test_metrics['precision']:.2%}): Percentage of alerts that are real threats")
        print(f"    -> {(1 - test_metrics['precision']) * 100:.1f}% of alerts are FALSE (False Positives)")

        # Print classification report
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=Config.CLASS_NAMES,
        ))

        return test_metrics

    def plot_confusion_matrix(self, test_metrics: Dict) -> None:
        """Generate and save confusion matrix plot."""
        cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"])

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=Config.CLASS_NAMES,
            yticklabels=Config.CLASS_NAMES,
            annot_kws={"size": 16},
        )
        plt.title("SUDARSHANchakra - Threat Detection Confusion Matrix", fontsize=14)
        plt.ylabel("Actual", fontsize=12)
        plt.xlabel("Predicted", fontsize=12)

        # Add defense context annotations
        plt.text(
            0.5, -0.15,
            f"False Negatives (Missed Threats): {cm[1, 0]} | False Positives (False Alarms): {cm[0, 1]}",
            ha="center",
            transform=plt.gca().transAxes,
            fontsize=10,
            color="red",
        )

        plt.tight_layout()
        save_path = Config.PLOTS_DIR / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n[SAVED] Confusion matrix: {save_path}")

    def plot_training_history(self) -> None:
        """Generate and save training history plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss plot
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(self.history["val_loss"], label="Val Loss", color="orange")
        axes[0, 0].set_title("Loss Over Epochs")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(self.history["train_acc"], label="Train Acc", color="blue")
        axes[0, 1].plot(self.history["val_acc"], label="Val Acc", color="orange")
        axes[0, 1].set_title("Accuracy Over Epochs")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision/Recall plot
        axes[1, 0].plot(self.history["val_precision"], label="Precision", color="green")
        axes[1, 0].plot(self.history["val_recall"], label="Recall", color="red")
        axes[1, 0].plot(self.history["val_f1"], label="F1", color="purple")
        axes[1, 0].set_title("Defense Metrics Over Epochs")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1, 1].plot(self.history["learning_rate"], color="gray")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("SUDARSHANchakra - Training Progress", fontsize=14)
        plt.tight_layout()

        save_path = Config.PLOTS_DIR / "training_history.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] Training history: {save_path}")

    def save_training_report(self, test_metrics: Dict) -> None:
        """Save comprehensive training report as JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sample_rate": Config.SAMPLE_RATE,
                "duration": Config.DURATION,
                "n_mels": Config.N_MELS,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "model_type": Config.MODEL_TYPE,
            },
            "final_metrics": {
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_f1": float(test_metrics["f1"]),
            },
            "training_history": {
                key: [float(v) for v in values]
                for key, values in self.history.items()
            },
        }

        save_path = Config.LOGS_DIR / "training_report.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[SAVED] Training report: {save_path}")


def run_training():
    """Execute complete training pipeline."""
    print("\n" + "=" * 60)
    print("SUDARSHANchakra - ACOUSTIC UAV THREAT DETECTION")
    print("Training Pipeline Initialization")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    # Get data loaders
    print("\n[STEP 1] Loading dataset...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders()

    # Create model
    print("\n[STEP 2] Initializing model...")
    model = get_model(Config.MODEL_TYPE)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
    )

    # Train
    print("\n[STEP 3] Starting training...")
    trainer.train()

    # Evaluate
    print("\n[STEP 4] Final evaluation...")
    test_metrics = trainer.evaluate()

    # Generate visualizations
    print("\n[STEP 5] Generating reports...")
    trainer.plot_confusion_matrix(test_metrics)
    trainer.plot_training_history()
    trainer.save_training_report(test_metrics)

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Best model saved at: {trainer.best_model_path}")
    print("=" * 60 + "\n")

    return trainer


if __name__ == "__main__":
    run_training()
