# Human Activity Recognition Classifier

A machine learning pipeline for classifying human activities (walking, sitting, standing, etc.) from smartphone accelerometer and gyroscope sensor data. Compares classical ML baselines against deep learning architectures, including domain adaptation and Bayesian uncertainty quantification.

## Datasets

- **KU-HAR** — Ku-Har dataset with 18 activity classes (filtered to 6 shared activities)
- **UCI HAR** — UCI Human Activity Recognition dataset (6 activities)

Both datasets use 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope) sampled at 50 Hz. Activities are harmonized to 6 shared classes: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying.

## Project Structure

```
human-activity-classifier/
├── src/
│   ├── config.py            # Hyperparameters, paths, experiment settings
│   ├── preprocessing.py     # Data loading, harmonization, windowing, splits
│   ├── features.py          # 176 hand-crafted feature extraction
│   ├── models.py            # Deep learning architectures (CNN, LSTM, TCN, etc.)
│   ├── classical_ml.py      # Classical ML baselines (LR, SVM, RF, XGBoost)
│   ├── train.py             # Training loops (standard + domain adaptation)
│   └── evaluate.py          # Metrics, confusion matrices, visualization
├── deep learning models.ipynb  # Notebook for running experiments
├── requirements.txt         # Python dependencies
└── README.md
```

## Models

### Classical ML (hand-crafted features)
- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest
- XGBoost (optional)

### Deep Learning (raw sensor windows)
- **CNN** — 1D convolutional network
- **LSTM** — Bidirectional LSTM
- **CNN+LSTM** — Hybrid convolutional-recurrent model
- **TCN** — Temporal Convolutional Network
- **TCN+DA** — TCN with Domain Adaptation (gradient reversal, Ganin et al.)
- **Bayesian CNN** — CNN with uncertainty quantification via Bayesian linear layers

## Experiments

| Experiment | Description |
|---|---|
| Within-dataset | Train and test on KU-HAR with subject-wise splits |
| Cross-dataset | Train on KU-HAR, test on UCI HAR (generalization) |
| Subject gap | Compare random vs. subject-wise splits to measure subject bias |
| Feature ablation | Evaluate contribution of time, frequency, and statistical feature groups |

## Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+

### Installation

```bash
pip install -r requirements.txt
```

### Data

Place the raw datasets in the `data/` directory:

```
data/
├── raw_kuhar/
│   └── 1.Raw_time_domian_data/
└── raw_ucihar/
    └── UCI HAR Dataset/
```

## Usage

Run experiments using the Jupyter notebook:

```bash
jupyter notebook "deep learning models.ipynb"
```

Or run the pipeline from the command line:

```python
from src.preprocessing import HARPreprocessor
from src.features import FeatureExtractor
from src.classical_ml import ClassicalMLModels
from src.train import train_model
from src.evaluate import ModelEvaluator

# Preprocess data
preprocessor = HARPreprocessor()
data = preprocessor.load_and_process()

# Classical ML
extractor = FeatureExtractor()
classical = ClassicalMLModels()

# Deep learning
train_model(model_name='TCN', data=data)

# Evaluate
evaluator = ModelEvaluator()
```

## Key Configuration

All hyperparameters are centralized in [src/config.py](src/config.py):

- Window size: 128 samples (2.56s at 50 Hz) with 50% overlap
- Batch size: 64
- Early stopping patience: 15 epochs
- Random seed: 42
