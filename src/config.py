"""
Configuration file for Human Activity Recognition Project
Contains all hyperparameters, paths, and experiment settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Raw data paths
KUHAR_RAW_PATH = DATA_ROOT / "raw_kuhar" / "1.Raw_time_domian_data"
UCIHAR_RAW_PATH = DATA_ROOT / "raw_ucihar" / "UCI HAR Dataset"

# Processed data paths
KUHAR_PROCESSED_PATH = DATA_ROOT / "processed_kuhar"
UCIHAR_PROCESSED_PATH = DATA_ROOT / "processed_ucihar"
COMBINED_PROCESSED_PATH = DATA_ROOT / "processed_combined"

# Results paths
FIGURES_PATH = RESULTS_ROOT / "figures"
MODELS_PATH = RESULTS_ROOT / "models"
METRICS_PATH = RESULTS_ROOT / "metrics"

# Ensure directories exist
for path in [FIGURES_PATH, MODELS_PATH, METRICS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Data processing parameters
WINDOW_SIZE = 128  # samples per window (2.56 seconds at 50Hz)
STRIDE = 64        # overlap (50% overlap)
SAMPLING_RATE = 50  # Hz

# Sensor channels
SENSOR_CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
N_CHANNELS = len(SENSOR_CHANNELS)

# Label harmonization - KU-HAR to UCI HAR mapping
# IMPORTANT: Only map activities that are truly equivalent across datasets.
# Mapping dissimilar activities (e.g. Run -> WALKING_UPSTAIRS) creates noisy
# labels that hurt deep learning more than classical ML.
SHARED_ACTIVITIES = {
    'WALKING': ['Walk'],
    'WALKING_UPSTAIRS': ['Stair-up'],
    'WALKING_DOWNSTAIRS': ['Stair-down'],
    'SITTING': ['Sit'],
    'STANDING': ['Stand'],
    'LAYING': ['Lay']
}

# Activities to drop from KU-HAR (no clean UCI equivalent)
KUHAR_ACTIVITIES_TO_DROP = [
    'Jump', 'Push-up', 'Pick', 'Sit-up', 'Table-tennis',
    'Lay-stand', 'Stand-sit', 'Run', 'Walk-backward', 'Walk-circle',
    'Talk-sit', 'Talk-stand'
]

# Final label mapping
LABEL_MAP = {
    'WALKING': 0,
    'WALKING_UPSTAIRS': 1, 
    'WALKING_DOWNSTAIRS': 2,
    'SITTING': 3,
    'STANDING': 4,
    'LAYING': 5
}
N_CLASSES = len(LABEL_MAP)

# Training hyperparameters
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
DEVICE = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'

# Model architectures
CNN_CONFIG = {
    'conv_layers': [32, 64, 128],
    'kernel_sizes': [5, 5, 5],
    'dropout': 0.4
}

LSTM_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': True,
    'dropout': 0.5
}

CNN_LSTM_CONFIG = {
    'conv_layers': [32, 64, 128],
    'kernel_sizes': [5, 5, 5],
    'lstm_hidden': 256,
    'lstm_layers': 2,
    'dropout': 0.5
}

TCN_CONFIG = {
    'channels': [64, 128, 256],
    'kernel_size': 3,
    'dropout': 0.3,
    'lstm_hidden': 128
}

# Domain adaptation parameters
DOMAIN_LAMBDA = 1.0   # max weight of domain confusion loss (Ganin default)
DOMAIN_LR = 1e-3      # learning rate for domain classifier
GRAD_REVERSAL_SCALE = 1.0

# Ganin schedule for lambda annealing
def ganin_lambda_schedule(p, max_lambda=DOMAIN_LAMBDA):
    """Anneal lambda from 0 to max_lambda over training (Ganin et al. ICML 2015)"""
    import math
    return (2.0 / (1.0 + math.exp(-10 * p)) - 1.0) * max_lambda

# Classical ML parameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 5

SVM_C = 1.0
SVM_KERNEL = 'rbf'

LR_C = 1.0
LR_PENALTY = 'l2'

# Feature extraction parameters
N_FEATURES = 176  # 96 time (16×6 axes) + 48 freq (8×6 axes) + 32 stat (16×2 sensors)

# Feature groups for ablation studies
# Time features are interleaved: acc_x(0-15), gyro_x(16-31), acc_y(32-47), gyro_y(48-63), acc_z(64-79), gyro_z(80-95)
# Freq features are interleaved: acc_x(96-103), gyro_x(104-111), acc_y(112-119), gyro_y(120-127), acc_z(128-135), gyro_z(136-143)
# Stat features are sequential: acc(144-159), gyro(160-175)
FEATURE_GROUPS = {
    'time_acc': list(range(0, 16)) + list(range(32, 48)) + list(range(64, 80)),
    'time_gyro': list(range(16, 32)) + list(range(48, 64)) + list(range(80, 96)),
    'freq_acc': list(range(96, 104)) + list(range(112, 120)) + list(range(128, 136)),
    'freq_gyro': list(range(104, 112)) + list(range(120, 128)) + list(range(136, 144)),
    'statistical_acc': list(range(144, 160)),
    'statistical_gyro': list(range(160, 176)),
}

# Experiment matrix
EXPERIMENTS = {
    'within_dataset': {
        'models': ['LR', 'SVM', 'RF', 'CNN', 'LSTM', 'CNN_LSTM', 'TCN', 'TCN_DA', 'BAYESIAN_CNN'],
        'train_dataset': 'kuhar',
        'test_dataset': 'kuhar',
        'split_type': 'subject'
    },
    'cross_dataset': {
        'models': ['RF', 'CNN_LSTM', 'TCN', 'TCN_DA', 'BAYESIAN_CNN'],
        'train_dataset': 'kuhar', 
        'test_dataset': 'ucihar',
        'split_type': 'subject'
    },
    'subject_gap': {
        'models': ['LR', 'RF', 'BAYESIAN_CNN'],
        'train_dataset': 'kuhar',
        'test_dataset': 'kuhar', 
        'split_types': ['random', 'subject']
    },
    'feature_ablation': {
        'models': ['LR', 'SVM', 'RF'],
        'feature_groups': list(FEATURE_GROUPS.keys()),
        'train_dataset': 'kuhar',
        'test_dataset': 'kuhar',
        'split_type': 'subject'
    }
}

# Evaluation metrics
METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Random seeds for reproducibility
RANDOM_SEED = 42
import numpy as np
import torch
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Visualization settings
PLOT_DPI = 300
FIGURE_SIZE = (10, 8)
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# File naming conventions
MODEL_CHECKPOINT_FMT = "{model}_{dataset}_{split_type}.pth"
CLASSICAL_MODEL_FMT = "{model}_{dataset}_{split_type}.pkl"
METRICS_CSV_FMT = "{model}_{dataset}_{split_type}_metrics.csv"
CONFUSION_MATRIX_FMT = "cm_{model}_{dataset}_{split_type}.png"
