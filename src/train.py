"""
Training loops for deep learning models
Supports both standard training and domain adaptation (DANN)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from config import *
from models import create_model, count_parameters
from evaluate import ModelEvaluator


def augment_batch(x, jitter_std=0.05, scale_range=(0.8, 1.2)):
    """On-the-fly augmentation for training batches.

    Args:
        x: tensor of shape (batch, channels, seq_len)
    Returns:
        augmented tensor (same shape)
    """
    x = x.clone()
    batch_size = x.size(0)

    # Jitter: add Gaussian noise
    x += torch.randn_like(x) * jitter_std

    # Scaling: multiply each sample by a random factor
    scales = torch.empty(batch_size, 1, 1, device=x.device).uniform_(*scale_range)
    x *= scales

    # Time shift: randomly shift each sample by up to 5 steps
    max_shift = 5
    for i in range(batch_size):
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift != 0:
            x[i] = torch.roll(x[i], shifts=shift, dims=-1)

    return x

class HARTrainer:
    """Trainer for HAR models with support for domain adaptation"""

    def __init__(self, model_name, device=DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.evaluator = ModelEvaluator()
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'domain_loss': [], 'domain_acc': []
        }
        self.target_loader = None  # For DANN

    def setup_model(self, model_config=None):
        """Initialize model and training components"""
        print(f"Setting up {self.model_name} model...")

        # Create model
        self.model = create_model(self.model_name, config=model_config)
        self.model = self.model.to(self.device)

        # Print model info
        param_count = count_parameters(self.model)
        print(f"Model: {self.model_name}")
        print(f"Parameters: {param_count:,}")
        print(f"Device: {self.device}")

        # Setup optimizer
        if self.model_name == "TCN_DA":
            # Different learning rates for encoder and classifiers
            self.optimizer = optim.Adam([
                {'params': self.model.encoder.parameters(), 'lr': LEARNING_RATE * 0.5},
                {'params': self.model.activity_classifier.parameters(), 'lr': LEARNING_RATE},
                {'params': self.model.domain_classifier.parameters(), 'lr': DOMAIN_LR}
            ], weight_decay=WEIGHT_DECAY)
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )

        # Setup loss functions
        self.criterion = nn.CrossEntropyLoss()

        # Bayesian training setup
        self.is_bayesian = "BAYESIAN" in self.model_name
        if self.is_bayesian:
            self.kl_weight = 0.001
            print(f"Bayesian model detected - KL weight: {self.kl_weight}")

        if self.model_name == "TCN_DA":
            self.domain_criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def prepare_data_loaders(self, X_train, y_train, X_val, y_val,
                             X_cross_test=None, y_cross_test=None,
                             X_target_domain=None):
        """Create data loaders, including target domain loader for DANN"""
        print("Preparing data loaders...")

        # Convert to tensors and transpose for CNN/TCN (channels first)
        X_train_tensor = torch.FloatTensor(X_train).transpose(1, 2)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).transpose(1, 2)
        y_val_tensor = torch.LongTensor(y_val)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create data loaders
        num_workers = 2 if torch.cuda.is_available() else 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=num_workers)

        # Prepare target domain loader for DANN (TCN_DA)
        if X_target_domain is not None and self.model_name == "TCN_DA":
            X_target_tensor = torch.FloatTensor(X_target_domain).transpose(1, 2)
            target_dataset = TensorDataset(X_target_tensor)
            self.target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=num_workers,
                                            drop_last=True)
            print(f"  Target domain loader: {len(X_target_domain)} samples "
                  f"({len(self.target_loader)} batches)")

        # Prepare cross-dataset data if available
        cross_loaders = {}
        if X_cross_test is not None and y_cross_test is not None:
            X_cross_tensor = torch.FloatTensor(X_cross_test).transpose(1, 2)
            y_cross_tensor = torch.LongTensor(y_cross_test)
            cross_dataset = TensorDataset(X_cross_tensor, y_cross_tensor)
            cross_loaders['test'] = DataLoader(cross_dataset, batch_size=BATCH_SIZE,
                                               shuffle=False)

        return train_loader, val_loader, cross_loaders

    def train_epoch(self, train_loader, epoch, lambda_val=None):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        domain_loss_total = 0
        domain_correct_total = 0
        domain_samples_total = 0

        # For TCN_DA, create an infinite iterator over target domain data
        target_iter = None
        if self.model_name == "TCN_DA" and self.target_loader is not None:
            target_iter = iter(self.target_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if self.model_name == "TCN_DA" and target_iter is not None:
                # ── DANN training: interleave source + target ──

                # Get a batch of target domain data (cycle if exhausted)
                try:
                    (target_data,) = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    (target_data,) = next(target_iter)
                target_data = target_data.to(self.device)

                # Match batch sizes (take smaller)
                min_batch = min(data.size(0), target_data.size(0))
                source_data = data[:min_batch]
                source_targets = targets[:min_batch]
                target_data = target_data[:min_batch]

                # Forward pass on SOURCE data — get activity + domain predictions
                source_activity_out, source_domain_out = self.model(
                    source_data, lambda_val=lambda_val
                )

                # Forward pass on TARGET data — get domain predictions only
                _, target_domain_out = self.model(
                    target_data, lambda_val=lambda_val
                )

                # Activity loss (source only — we have labels)
                activity_loss = self.criterion(source_activity_out, source_targets)

                # Domain loss: source=0, target=1
                source_domain_labels = torch.zeros(min_batch, dtype=torch.long,
                                                   device=self.device)
                target_domain_labels = torch.ones(min_batch, dtype=torch.long,
                                                  device=self.device)

                domain_loss_source = self.domain_criterion(
                    source_domain_out, source_domain_labels
                )
                domain_loss_target = self.domain_criterion(
                    target_domain_out, target_domain_labels
                )
                domain_loss = (domain_loss_source + domain_loss_target) / 2.0

                # Combined loss (GRL handles sign flip for encoder)
                total_batch_loss = activity_loss + domain_loss

                total_batch_loss.backward()
                self.optimizer.step()

                # Metrics
                total_loss += activity_loss.item()
                domain_loss_total += domain_loss.item()

                _, predicted = torch.max(source_activity_out.data, 1)
                total_correct += (predicted == source_targets).sum().item()
                total_samples += source_targets.size(0)

                # Domain accuracy (should trend toward 50% = confused)
                all_domain_out = torch.cat([source_domain_out, target_domain_out])
                all_domain_labels = torch.cat([source_domain_labels, target_domain_labels])
                _, domain_predicted = torch.max(all_domain_out.data, 1)
                domain_correct_total += (domain_predicted == all_domain_labels).sum().item()
                domain_samples_total += all_domain_labels.size(0)

                progress_bar.set_postfix({
                    'ActLoss': f"{activity_loss.item():.4f}",
                    'DomLoss': f"{domain_loss.item():.4f}",
                    'Acc': f"{100. * total_correct / total_samples:.1f}%",
                    'DomAcc': f"{100. * domain_correct_total / domain_samples_total:.1f}%"
                })

            elif self.is_bayesian:
                # Bayesian training with KL divergence
                outputs = self.model(data)
                classification_loss = self.criterion(outputs, targets)
                kl_loss = self.model.kl_divergence()
                total_batch_loss = classification_loss + self.kl_weight * kl_loss

                total_batch_loss.backward()
                self.optimizer.step()

                total_loss += classification_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                progress_bar.set_postfix({
                    'Loss': f"{classification_loss.item():.4f}",
                    'Acc': f"{100. * total_correct / total_samples:.1f}%"
                })

            else:
                # Standard training
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100. * total_correct / total_samples:.1f}%"
                })

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * total_correct / total_samples

        if self.model_name == "TCN_DA" and domain_samples_total > 0:
            avg_domain_loss = domain_loss_total / len(train_loader)
            domain_accuracy = 100. * domain_correct_total / domain_samples_total
            return avg_loss, accuracy, avg_domain_loss, domain_accuracy
        else:
            return avg_loss, accuracy, None, None

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                if self.model_name == "TCN_DA":
                    activity_output, _ = self.model(data)
                else:
                    activity_output = self.model(data)

                loss = self.criterion(activity_output, targets)
                total_loss += loss.item()

                _, predicted = torch.max(activity_output.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * total_correct / total_samples

        return avg_loss, accuracy, all_predictions, all_targets

    def train(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None,
              X_cross_test=None, y_cross_test=None, X_target_domain=None):
        """Main training loop.

        Args:
            X_target_domain: Unlabeled target domain data for DANN (TCN_DA only).
                             Shape: (n_samples, 128, 6). Required for TCN_DA.
        """
        print(f"Training {self.model_name}...")

        # Warn if TCN_DA is used without target domain data
        if self.model_name == "TCN_DA" and X_target_domain is None:
            print("WARNING: TCN_DA requires X_target_domain for domain adaptation!")
            print("         Without it, DANN cannot learn domain-invariant features.")

        # Prepare data
        train_loader, val_loader, cross_loaders = self.prepare_data_loaders(
            X_train, y_train, X_val, y_val, X_cross_test, y_cross_test,
            X_target_domain=X_target_domain
        )

        # Prepare test loader if available
        test_loader = None
        if X_test is not None and y_test is not None:
            X_test_tensor = torch.FloatTensor(X_test).transpose(1, 2)
            y_test_tensor = torch.LongTensor(y_test)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training loop
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            # Calculate lambda for domain adaptation (Ganin annealing)
            if self.model_name == "TCN_DA":
                p = epoch / MAX_EPOCHS
                lambda_val = ganin_lambda_schedule(p)
                self.model.set_domain_lambda(lambda_val)
            else:
                lambda_val = None

            # Train
            train_loss, train_acc, domain_loss, domain_acc = self.train_epoch(
                train_loader, epoch, lambda_val
            )

            # Validate
            val_loss, val_acc, val_pred, val_true = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            if self.model_name == "TCN_DA" and domain_loss is not None:
                self.training_history['domain_loss'].append(domain_loss)
                self.training_history['domain_acc'].append(domain_acc)

            # Print epoch summary
            print(f"Epoch {epoch+1}/{MAX_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if self.model_name == "TCN_DA" and domain_loss is not None:
                print(f"  Domain Loss: {domain_loss:.4f}, Domain Acc: {domain_acc:.2f}%")
                print(f"  Lambda: {lambda_val:.4f}")

            # Early stopping
            # For TCN_DA: don't allow early stopping until epoch 50 so that
            # the Ganin lambda schedule has fully ramped up and domain
            # adaptation has had time to align source/target distributions.
            min_epochs = 50 if self.model_name == "TCN_DA" else 0

            if epoch >= min_epochs and val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save best model (for TCN_DA, only after domain adaptation has ramped up)
                torch.save(self.model.state_dict(), MODELS_PATH / f"{self.model_name}_best.pth")
                print(f"  New best model saved! Val Acc: {val_acc:.2f}%")
            elif epoch >= min_epochs:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                print(f"  No improvement for {patience_counter} epochs")
            else:
                # Before min_epochs: always save latest as checkpoint
                # (so we have something to load if training crashes)
                torch.save(self.model.state_dict(), MODELS_PATH / f"{self.model_name}_best.pth")
                print(f"  Warmup epoch {epoch+1}/{min_epochs} (no early stopping yet)")

        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(MODELS_PATH / f"{self.model_name}_best.pth",
                                              weights_only=True))

        # Final evaluation on validation set
        print(f"\nFinal evaluation on validation set:")
        val_loss, val_acc, val_pred, val_true = self.validate_epoch(val_loader)
        val_metrics = self.evaluator.calculate_metrics(val_true, val_pred)

        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")

        # Held-out test set evaluation (KU-HAR, subject-independent)
        test_metrics = {}
        if test_loader is not None:
            print(f"\nFinal evaluation on KU-HAR held-out test set:")
            test_loss, test_acc, test_pred, test_true = self.validate_epoch(test_loader)
            test_metrics = self.evaluator.calculate_metrics(test_true, test_pred)
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")

        # Cross-dataset evaluation if available
        cross_results = {}
        if cross_loaders:
            print(f"\nCross-dataset evaluation (UCI HAR):")
            for split_name, loader in cross_loaders.items():
                cross_loss, cross_acc, cross_pred, cross_true = self.validate_epoch(loader)
                cross_metrics = self.evaluator.calculate_metrics(cross_true, cross_pred)
                cross_results[split_name] = cross_metrics

                print(f"{split_name} Accuracy: {cross_acc:.2f}%")
                print(f"{split_name} F1 (macro): {cross_metrics['f1_macro']:.4f}")

        # Save training history
        self.save_training_history()

        # Save final model
        torch.save(self.model.state_dict(), MODELS_PATH / f"{self.model_name}_final.pth")

        return val_metrics, test_metrics, cross_results

    def save_training_history(self):
        """Save training history"""
        history_path = METRICS_PATH / f"{self.model_name}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Plot learning curves
        self.plot_learning_curves()

    def plot_learning_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.training_history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Domain-specific curves (for TCN_DA)
        if self.model_name == "TCN_DA" and self.training_history.get('domain_loss'):
            axes[1, 0].plot(self.training_history['domain_loss'], label='Domain Loss')
            axes[1, 0].set_title('Domain Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            axes[1, 1].plot(self.training_history['domain_acc'], label='Domain Acc')
            axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
            axes[1, 1].set_title('Domain Accuracy (should approach 50%)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)

        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f"{self.model_name}_learning_curves.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

def train_all_models():
    """Train all deep learning models"""
    print("=== Training Deep Learning Models ===")

    # Load data
    X_train = np.load(COMBINED_PROCESSED_PATH / "X_train.npy")
    y_train = np.load(COMBINED_PROCESSED_PATH / "y_train.npy")
    X_val = np.load(COMBINED_PROCESSED_PATH / "X_val.npy")
    y_val = np.load(COMBINED_PROCESSED_PATH / "y_val.npy")
    X_test = np.load(COMBINED_PROCESSED_PATH / "X_test.npy")
    y_test = np.load(COMBINED_PROCESSED_PATH / "y_test.npy")

    # Load cross-dataset test if available
    X_cross_test = None
    y_cross_test = None
    X_target_domain = None
    cross_available = False

    try:
        X_cross_test = np.load(COMBINED_PROCESSED_PATH / "X_cross_test.npy")
        y_cross_test = np.load(COMBINED_PROCESSED_PATH / "y_cross_test.npy")
        cross_available = True
    except FileNotFoundError:
        print("Cross-dataset test data not available")

    try:
        X_target_domain = np.load(COMBINED_PROCESSED_PATH / "X_target_domain.npy")
        print(f"Target domain data loaded: {X_target_domain.shape}")
    except FileNotFoundError:
        print("Target domain data not available (needed for TCN_DA)")

    # Models to train
    if cross_available:
        models_to_train = ["CNN", "LSTM", "CNN_LSTM", "TCN", "TCN_DA"]
    else:
        models_to_train = ["CNN", "LSTM", "CNN_LSTM", "TCN"]

    all_results = {}

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        trainer = HARTrainer(model_name)
        trainer.setup_model()

        # Only pass target domain data to TCN_DA
        target_data = X_target_domain if model_name == "TCN_DA" else None

        val_metrics, test_metrics, cross_results = trainer.train(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_cross_test, y_cross_test,
            X_target_domain=target_data
        )

        all_results[model_name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cross_dataset': cross_results
        }

        print(f"{model_name} training completed!")

    # Save all results
    results_df = pd.DataFrame({
        model: {
            'val_accuracy': results['val_metrics'].get('accuracy', None),
            'val_f1_macro': results['val_metrics'].get('f1_macro', None),
            'test_accuracy': results['test_metrics'].get('accuracy', None),
            'test_f1_macro': results['test_metrics'].get('f1_macro', None),
            'cross_accuracy': results['cross_dataset'].get('test', {}).get('accuracy', None),
            'cross_f1_macro': results['cross_dataset'].get('test', {}).get('f1_macro', None)
        }
        for model, results in all_results.items()
    }).T

    results_df.to_csv(METRICS_PATH / "deep_learning_metrics.csv")

    print(f"\n{'='*60}")
    print("Deep Learning Results Summary")
    print(f"{'='*60}")
    print(results_df.round(4))

    return all_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument("--model", choices=["CNN", "LSTM", "CNN_LSTM", "TCN", "TCN_DA", "all"],
                       default="all", help="Which model to train")
    parser.add_argument("--input_type", choices=["timeseries", "features"],
                       default="timeseries", help="Input data type")

    args = parser.parse_args()

    if args.model == "all":
        train_all_models()
    else:
        print(f"Training single model: {args.model}")

        X_train = np.load(COMBINED_PROCESSED_PATH / "X_train.npy")
        y_train = np.load(COMBINED_PROCESSED_PATH / "y_train.npy")
        X_val = np.load(COMBINED_PROCESSED_PATH / "X_val.npy")
        y_val = np.load(COMBINED_PROCESSED_PATH / "y_val.npy")
        X_test = np.load(COMBINED_PROCESSED_PATH / "X_test.npy")
        y_test = np.load(COMBINED_PROCESSED_PATH / "y_test.npy")

        X_cross_test = None
        y_cross_test = None
        X_target_domain = None

        try:
            X_cross_test = np.load(COMBINED_PROCESSED_PATH / "X_cross_test.npy")
            y_cross_test = np.load(COMBINED_PROCESSED_PATH / "y_cross_test.npy")
        except FileNotFoundError:
            pass

        if args.model == "TCN_DA":
            try:
                X_target_domain = np.load(COMBINED_PROCESSED_PATH / "X_target_domain.npy")
            except FileNotFoundError:
                print("WARNING: X_target_domain.npy not found!")

        trainer = HARTrainer(args.model)
        trainer.setup_model()
        val_metrics, test_metrics, cross_results = trainer.train(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_cross_test, y_cross_test,
            X_target_domain=X_target_domain
        )

        print(f"Val results: {val_metrics}")
        print(f"Test results: {test_metrics}")
        if cross_results:
            print(f"Cross results: {cross_results}")
