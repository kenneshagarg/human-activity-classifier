"""
Evaluation and visualization module for Human Activity Recognition
Handles metrics calculation, confusion matrices, and result visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import os
from typing import Dict, List, Tuple, Any

from config import *

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self):
        self.label_map = LABEL_MAP
        self.reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true, y_pred, average='macro'):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist()
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, dataset_type="within", normalize=True):
        """Plot and save confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        plt.figure(figsize=FIGURE_SIZE)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=list(LABEL_MAP.keys()),
                   yticklabels=list(LABEL_MAP.keys()))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        filename = f"cm_{model_name}_{dataset_type}.png"
        plt.savefig(FIGURES_PATH / filename, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_class_distribution(self, y_data, split_name, dataset_name="KU-HAR"):
        """Plot class distribution"""
        
        # Convert labels to names
        label_names = [self.reverse_label_map[label] for label in y_data]
        
        plt.figure(figsize=FIGURE_SIZE)
        ax = sns.countplot(x=label_names, order=list(LABEL_MAP.keys()))
        plt.title(f'Class Distribution - {dataset_name} {split_name}')
        plt.xlabel('Activity')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f"class_distribution_{split_name.lower()}.png", 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_signal_samples(self, X_data, y_data, n_samples_per_class=2):
        """Plot sample sensor signals for each activity"""
        
        plt.figure(figsize=(15, 12))
        
        for class_idx, activity in enumerate(LABEL_MAP.keys()):
            # Find samples for this class
            class_mask = y_data == class_idx
            class_samples = X_data[class_mask]
            
            if len(class_samples) == 0:
                continue
            
            # Select random samples
            n_samples = min(n_samples_per_class, len(class_samples))
            sample_indices = np.random.choice(len(class_samples), n_samples, replace=False)
            
            for sample_idx in range(n_samples):
                sample = class_samples[sample_indices[sample_idx]]
                
                # Create subplot for this sample
                subplot_idx = class_idx * n_samples_per_class + sample_idx + 1
                plt.subplot(len(LABEL_MAP), n_samples_per_class, subplot_idx)
                
                # Plot each sensor channel
                time_axis = np.arange(len(sample)) / SAMPLING_RATE
                
                for channel_idx, channel_name in enumerate(['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']):
                    offset = channel_idx * 2  # Separate channels vertically
                    plt.plot(time_axis, sample[:, channel_idx] + offset, 
                           label=channel_name, alpha=0.7)
                
                plt.title(f'{activity} - Sample {sample_idx + 1}')
                plt.xlabel('Time (s)')
                plt.ylabel('Normalized Value')
                if subplot_idx == 1:  # Only show legend for first subplot
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "signal_samples.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_sensor_correlation(self, X_data, dataset_name="KU-HAR"):
        """Plot correlation heatmap of sensor channels"""
        
        # Reshape data to (n_samples, n_channels)
        n_samples, seq_len, n_channels = X_data.shape
        X_reshaped = X_data.reshape(-1, n_channels)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_reshaped.T)
        
        plt.figure(figsize=FIGURE_SIZE)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                   yticklabels=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        plt.title(f'Sensor Channel Correlation - {dataset_name}')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "sensor_correlation.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_pca_analysis(self, X_data, y_data, dataset_name="KU-HAR"):
        """Perform and visualize PCA analysis"""
        
        # Reshape for PCA
        n_samples, seq_len, n_channels = X_data.shape
        X_reshaped = X_data.reshape(n_samples, -1)
        
        # Perform PCA
        pca = PCA(n_components=min(50, X_reshaped.shape[1]))
        X_pca = pca.fit_transform(X_reshaped)
        
        # Plot explained variance
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'PCA Explained Variance - {dataset_name}')
        plt.legend()
        plt.grid(True)
        
        # Plot 2D PCA scatter
        plt.subplot(1, 2, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(LABEL_MAP)))
        
        for class_idx, activity in enumerate(LABEL_MAP.keys()):
            class_mask = y_data == class_idx
            if np.any(class_mask):
                plt.scatter(X_pca[class_mask, 0], X_pca[class_mask, 1], 
                          c=[colors[class_idx]], label=activity, alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
        plt.title(f'2D PCA Visualization - {dataset_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "pca_analysis.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return pca, X_pca
    
    def plot_tsne(self, X_data, y_data, dataset_name="KU-HAR", perplexity=30):
        """Perform and visualize t-SNE analysis"""
        
        # Reshape for t-SNE
        n_samples, seq_len, n_channels = X_data.shape
        X_reshaped = X_data.reshape(n_samples, -1)
        
        # Subsample if too many samples
        if n_samples > 5000:
            indices = np.random.choice(n_samples, 5000, replace=False)
            X_subset = X_reshaped[indices]
            y_subset = y_data[indices]
        else:
            X_subset = X_reshaped
            y_subset = y_data
        
        # Perform t-SNE
        print(f"Running t-SNE on {len(X_subset)} samples...")
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X_subset)//3), 
                   random_state=RANDOM_SEED, n_iter=1000)
        X_tsne = tsne.fit_transform(X_subset)
        
        # Plot t-SNE visualization
        plt.figure(figsize=FIGURE_SIZE)
        colors = plt.cm.Set3(np.linspace(0, 1, len(LABEL_MAP)))
        
        for class_idx, activity in enumerate(LABEL_MAP.keys()):
            class_mask = y_subset == class_idx
            if np.any(class_mask):
                plt.scatter(X_tsne[class_mask, 0], X_tsne[class_mask, 1], 
                          c=[colors[class_idx]], label=activity, alpha=0.7)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE Visualization - {dataset_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "tsne.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return tsne, X_tsne
    
    def plot_kmeans_analysis(self, X_data, y_data, dataset_name="KU-HAR", max_k=10):
        """Analyze clustering quality with K-means"""
        
        # Reshape for clustering
        n_samples, seq_len, n_channels = X_data.shape
        X_reshaped = X_data.reshape(n_samples, -1)
        
        # Test different numbers of clusters
        k_range = range(2, min(max_k + 1, len(X_reshaped)))
        nmi_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            cluster_labels = kmeans.fit_predict(X_reshaped)
            
            # Calculate Normalized Mutual Information
            from sklearn.metrics import normalized_mutual_info_score
            nmi = normalized_mutual_info_score(y_data, cluster_labels)
            nmi_scores.append(nmi)
            
            print(f"K={k}: NMI={nmi:.4f}")
        
        # Plot NMI vs K
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(k_range, nmi_scores, 'bo-')
        plt.axvline(x=len(LABEL_MAP), color='r', linestyle='--', 
                   label=f'True number of classes ({len(LABEL_MAP)})')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Normalized Mutual Information')
        plt.title(f'K-means Clustering Quality - {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "kmeans_nmi_elbow.png", dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return k_range, nmi_scores
    
    def plot_model_comparison(self, results_dict, metric='f1_macro'):
        """Compare model performance across different conditions"""
        
        # Prepare data for plotting
        models = []
        conditions = []
        values = []
        
        for model, model_results in results_dict.items():
            if isinstance(model_results, dict):
                for condition, metrics in model_results.items():
                    if isinstance(metrics, dict) and metric in metrics:
                        models.append(model)
                        conditions.append(condition)
                        values.append(metrics[metric])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Model': models,
            'Condition': conditions,
            'Value': values
        })
        
        # Plot
        plt.figure(figsize=FIGURE_SIZE)
        sns.barplot(data=df, x='Model', y='Value', hue='Condition')
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Condition')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f"model_comparison_{metric}.png", 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return df
    
    def plot_cross_dataset_comparison(self, within_results, cross_results, model_names):
        """Plot side-by-side comparison of within vs cross-dataset performance"""
        
        metrics = ['accuracy', 'f1_macro']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, metric in enumerate(metrics):
            within_values = []
            cross_values = []
            
            for model in model_names:
                if model in within_results and metric in within_results[model]:
                    within_values.append(within_results[model][metric])
                else:
                    within_values.append(0)
                
                if model in cross_results and metric in cross_results[model]:
                    cross_values.append(cross_results[model][metric])
                else:
                    cross_values.append(0)
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[idx].bar(x - width/2, within_values, width, label='Within Dataset', alpha=0.8)
            axes[idx].bar(x + width/2, cross_values, width, label='Cross Dataset', alpha=0.8)
            
            axes[idx].set_xlabel('Model')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(model_names, rotation=45)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "cross_dataset_comparison.png", 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_subject_independence_gap(self, random_results, subject_results):
        """Plot the performance gap between random and subject splits"""
        
        models = list(random_results.keys())
        random_acc = [random_results[model]['accuracy'] for model in models]
        subject_acc = [subject_results[model]['accuracy'] for model in models]
        
        gaps = [r - s for r, s in zip(random_acc, subject_acc)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot comparison
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, random_acc, width, label='Random Split', alpha=0.8)
        ax1.bar(x + width/2, subject_acc, width, label='Subject Split', alpha=0.8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Random vs Subject Split Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gap plot
        colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in gaps]
        ax2.bar(models, gaps, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Large Gap (>0.1)')
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Medium Gap (>0.05)')
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Accuracy Gap')
        ax2.set_title('Subject Independence Gap (Random - Subject)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "subject_independence_gap.png", 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        return gaps
    
    def generate_latex_table(self, results_dict, caption="Model Performance Comparison"):
        """Generate LaTeX table for results"""
        
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{tab:results}",
            "\\begin{tabular}{lcc}",
            "\\hline",
            "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1-Score} \\\\",
            "\\hline"
        ]
        
        for model, metrics in results_dict.items():
            if isinstance(metrics, dict):
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_macro', 0)
                latex_lines.append(f"{model} & {acc:.4f} & {f1:.4f} \\\\")
        
        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        latex_table = "\n".join(latex_lines)
        
        # Save to file
        with open(METRICS_PATH / "results_table.tex", 'w') as f:
            f.write(latex_table)
        
        return latex_table
    
    def save_metrics_json(self, metrics_dict, filename="metrics.json"):
        """Save metrics to JSON file"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_metrics = convert_numpy(metrics_dict)
        
        with open(METRICS_PATH / filename, 'w') as f:
            json.dump(converted_metrics, f, indent=2)

def run_eda_pipeline():
    """Run complete Exploratory Data Analysis pipeline"""
    print("=== Running EDA Pipeline ===")
    
    # Load data
    try:
        X_train = np.load(COMBINED_PROCESSED_PATH / "X_train.npy")
        y_train = np.load(COMBINED_PROCESSED_PATH / "y_train.npy")
        X_val = np.load(COMBINED_PROCESSED_PATH / "X_val.npy")
        y_val = np.load(COMBINED_PROCESSED_PATH / "y_val.npy")
        X_test = np.load(COMBINED_PROCESSED_PATH / "X_test.npy")
        y_test = np.load(COMBINED_PROCESSED_PATH / "y_test.npy")
        
        # Combine train and val for some analyses
        X_all = np.vstack([X_train, X_val])
        y_all = np.hstack([y_train, y_val])
        
        print(f"Data loaded: X_train={X_train.shape}, y_train={y_train.shape}")
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please run preprocessing first!")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # 1. Class distribution
    print("Plotting class distributions...")
    evaluator.plot_class_distribution(y_train, "Train")
    evaluator.plot_class_distribution(y_test, "Test")
    
    # 2. Signal samples
    print("Plotting signal samples...")
    evaluator.plot_signal_samples(X_train, y_train)
    
    # 3. Sensor correlation
    print("Plotting sensor correlations...")
    evaluator.plot_sensor_correlation(X_train)
    
    # 4. PCA analysis
    print("Running PCA analysis...")
    evaluator.plot_pca_analysis(X_train, y_train)
    
    # 5. t-SNE visualization
    print("Running t-SNE...")
    evaluator.plot_tsne(X_train, y_train)
    
    # 6. K-means clustering analysis
    print("Running K-means analysis...")
    evaluator.plot_kmeans_analysis(X_train, y_train)
    
    print("EDA pipeline complete!")
    print(f"All plots saved to {FIGURES_PATH}")

if __name__ == "__main__":
    run_eda_pipeline()
