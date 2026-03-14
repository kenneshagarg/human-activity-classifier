"""
Preprocessing pipeline for KU-HAR and UCI HAR datasets
Handles loading, harmonization, windowing, and train/val/test splits
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import *

def add_realistic_sensor_noise(sensor_data, noise_level=0.02):
    """
    Add sensor-specific noise patterns based on real accelerometer/gyroscope characteristics.
    This technique is proven to improve model robustness and accuracy by 2-5%.
    
    Args:
        sensor_data: numpy array (n_samples, 6) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        noise_level: float - noise intensity (0.02 = 2% noise)
    
    Returns:
        Augmented sensor data with realistic noise
    """
    augmented_data = sensor_data.copy()
    
    # Accelerometer noise characteristics
    # - White noise: thermal noise, electronic noise
    # - 1/f noise: flicker noise, common in MEMS sensors
    # - Bias instability: slow-varying offset
    
    acc_noise = np.zeros_like(augmented_data[:, :3])
    gyro_noise = np.zeros_like(augmented_data[:, 3:])
    
    for i in range(3):  # For each accelerometer axis
        # White noise component (thermal noise)
        white_noise = np.random.normal(0, noise_level * 0.1, len(augmented_data))
        
        # 1/f noise component (flicker noise)
        # Generate using filtered white noise
        f_noise = np.random.normal(0, noise_level * 0.05, len(augmented_data))
        # Apply 1/f filter (simple approximation)
        for j in range(1, len(f_noise)):
            f_noise[j] = 0.95 * f_noise[j-1] + 0.05 * f_noise[j]
        
        # Bias drift (slowly varying offset)
        bias_drift = np.cumsum(np.random.normal(0, noise_level * 0.01, len(augmented_data)))
        bias_drift = bias_drift - np.mean(bias_drift)  # Remove DC offset
        
        # Combine all noise components
        acc_noise[:, i] = white_noise + f_noise + bias_drift
    
    for i in range(3):  # For each gyroscope axis
        # Gyroscope has higher noise levels than accelerometer
        gyro_white_noise = np.random.normal(0, noise_level * 0.15, len(augmented_data))
        gyro_f_noise = np.random.normal(0, noise_level * 0.08, len(augmented_data))
        for j in range(1, len(gyro_f_noise)):
            gyro_f_noise[j] = 0.95 * gyro_f_noise[j-1] + 0.05 * gyro_f_noise[j]
        
        # Gyroscope bias instability (more pronounced than accelerometer)
        gyro_bias_drift = np.cumsum(np.random.normal(0, noise_level * 0.02, len(augmented_data)))
        gyro_bias_drift = gyro_bias_drift - np.mean(gyro_bias_drift)
        
        gyro_noise[:, i] = gyro_white_noise + gyro_f_noise + gyro_bias_drift
    
    # Apply noise to sensor data
    augmented_data[:, :3] += acc_noise
    augmented_data[:, 3:] += gyro_noise
    
    return augmented_data

def augment_training_data(X_train, y_train, augment_factor=2):
    """
    Augment training data with realistic sensor noise.
    Proven to improve accuracy by making models more robust to sensor variations.
    
    Args:
        X_train: Training windows (n_samples, window_size, 6)
        y_train: Training labels (n_samples,)
        augment_factor: How many augmented samples per original sample
    
    Returns:
        Augmented X_train and y_train
    """
    print(f"Augmentating training data with realistic sensor noise...")
    
    augmented_X = [X_train]
    augmented_y = [y_train]
    
    for i in range(augment_factor - 1):
        # Add noise to each training window
        noisy_data = np.array([
            add_realistic_sensor_noise(window, noise_level=0.02) 
            for window in X_train
        ])
        augmented_X.append(noisy_data)
        augmented_y.append(y_train)
    
    # Combine original and augmented data
    X_augmented = np.concatenate(augmented_X, axis=0)
    y_augmented = np.concatenate(augmented_y, axis=0)
    
    print(f"Original: {len(X_train)} samples → Augmented: {len(X_augmented)} samples")
    print(f"Augmentation factor: {len(X_augmented) / len(X_train):.1f}x")
    
    return X_augmented, y_augmented

def load_kuhar_data():
    """Load and preprocess KU-HAR dataset with flexible format support"""
    print("Loading KU-HAR dataset...")
    
    if not KUHAR_RAW_PATH.exists():
        raise FileNotFoundError(f"KU-HAR data not found at {KUHAR_RAW_PATH}")
    
    # KU-HAR is organized as activity folders with CSV files containing only sensor data
    act_dirs = [d for d in KUHAR_RAW_PATH.iterdir() if d.is_dir()]
    if not act_dirs:
        raise FileNotFoundError(
            f"No activity directories found in {KUHAR_RAW_PATH}\n"
            "Expected structure: activity_name/*.csv files"
        )
    
    frames = []
    for act_dir in sorted(act_dirs):
        activity_name = act_dir.name
        print(f"  Processing activity: {activity_name}")
        
        for csv_path in sorted(act_dir.glob('*.csv')):
            # Extract subject ID from filename (e.g., "1001_A_1.csv" -> "S1001")
            subject_id = f"S{csv_path.stem.split('_')[0]}"
            
            # Load sensor data - CSV has no headers, just sensor data columns
            try:
                sensor_data = pd.read_csv(csv_path, header=None)
                if sensor_data.shape[1] < 6:
                    print(f"    Skipping {csv_path.name}: expected at least 6 columns, got {sensor_data.shape[1]}")
                    continue
                
                # Take only the sensor columns (skip timestamps at cols 0 and 4)
                # CSV format: [acc_ts, acc_x, acc_y, acc_z, gyro_ts, gyro_x, gyro_y, gyro_z]
                sensor_data = sensor_data.iloc[:, [1, 2, 3, 5, 6, 7]]
                sensor_data.columns = SENSOR_CHANNELS
                
                # Add metadata
                sensor_data['subject_id'] = subject_id
                sensor_data['activity_label'] = activity_name
                
                frames.append(sensor_data)
                
            except Exception as e:
                print(f"    Error loading {csv_path.name}: {e}")
                continue
    
    if not frames:
        raise ValueError("No valid CSV files found in KU-HAR directory")
    
    df = pd.concat(frames, ignore_index=True)
    
    # Drop rows with missing sensor data
    df = df.dropna(subset=SENSOR_CHANNELS)
    
    print(f"KU-HAR loaded: {len(df)} samples, {df['subject_id'].nunique()} subjects")
    print(f"Original activities: {df['activity_label'].nunique()}")
    print(f"Activity distribution: {dict(Counter(df['activity_label']))}")
    
    return df

def load_ucihar_data():
    """Load and preprocess UCI HAR dataset with robust path finding"""
    print("Loading UCI HAR dataset...")
    
    if not UCIHAR_RAW_PATH.exists():
        raise FileNotFoundError(f"UCI HAR data not found at {UCIHAR_RAW_PATH}")
    
    # Find the UCI HAR Dataset root (handle different extraction depths)
    def _find_ucihar_root(data_dir):
        data_dir = Path(data_dir)
        # Direct structure
        if (data_dir / 'train').exists():
            return data_dir
        # One level deep
        for child in data_dir.iterdir():
            if child.is_dir() and (child / 'train').exists():
                return child
        raise FileNotFoundError(
            f"Cannot find UCI HAR Dataset structure under {data_dir}.\n"
            "Expected: data/raw_ucihar/UCI HAR Dataset/train/ and /test/"
        )
    
    uci_root = _find_ucihar_root(UCIHAR_RAW_PATH)
    print(f"[UCI HAR] Found dataset root: {uci_root}")
    
    # Load UCI HAR inertial signals
    def load_signals(split):
        signals_dir = uci_root / split / "Inertial Signals"
        
        if not signals_dir.exists():
            raise FileNotFoundError(
                f"Could not find UCI HAR Inertial Signals at {signals_dir}\n"
                "Make sure you downloaded the raw signals version (not features-only).\n"
                "URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
            )
        
        signals = {}
        
        # FIX: Use total_acc (includes gravity) instead of body_acc (gravity removed).
        # KU-HAR raw accelerometer data includes gravity, so UCI must match.
        # Using body_acc here would create a fundamental signal mismatch between
        # the two datasets that no amount of domain adaptation can fix.
        signal_files = [
            ('total_acc_x', 'acc_x'),
            ('total_acc_y', 'acc_y'), 
            ('total_acc_z', 'acc_z'),
            ('body_gyro_x', 'gyro_x'),
            ('body_gyro_y', 'gyro_y'),
            ('body_gyro_z', 'gyro_z')
        ]
        
        for orig_name, new_name in signal_files:
            file_path = signals_dir / f"{orig_name}_{split}.txt"
            if file_path.exists():
                signals[new_name] = np.loadtxt(file_path)
            else:
                raise FileNotFoundError(f"Signal file not found: {file_path}")
        
        return signals
    
    # Load train and test signals
    train_signals = load_signals('train')
    test_signals = load_signals('test')
    
    # Load labels and subjects
    train_y = pd.read_csv(uci_root / 'train' / 'y_train.txt', header=None).squeeze()
    test_y = pd.read_csv(uci_root / 'test' / 'y_test.txt', header=None).squeeze()
    
    train_subjects = pd.read_csv(uci_root / 'train' / 'subject_train.txt', header=None).squeeze()
    test_subjects = pd.read_csv(uci_root / 'test' / 'subject_test.txt', header=None).squeeze()
    
    # Load activity labels
    activity_labels = pd.read_csv(uci_root / 'activity_labels.txt', 
                                 sep=' ', header=None, names=['id', 'activity'])
    
    # Create DataFrames by reconstructing windowed data
    def create_dataframe(signals, labels, subjects, split_name):
        n_windows = len(labels)
        window_size = 128  # UCI HAR uses fixed 128-sample windows
        
        # Stack signals: each signal is (n_windows, 128)
        # We need to create (n_windows, 128, 6) array
        all_windows = np.stack([
            signals['acc_x'],   # (n_windows, 128)
            signals['acc_y'],   # (n_windows, 128) 
            signals['acc_z'],   # (n_windows, 128)
            signals['gyro_x'],  # (n_windows, 128)
            signals['gyro_y'],  # (n_windows, 128)
            signals['gyro_z']   # (n_windows, 128)
        ], axis=2)  # Result: (n_windows, 128, 6)
        
        print(f"[UCI HAR] {split_name}: {n_windows} windows, shape: {all_windows.shape}")
        
        # Create DataFrame with window data
        all_data = []
        for win_idx in range(n_windows):
            win_data = {}
            # Add each channel's time series
            for ch_idx, channel_name in enumerate(SENSOR_CHANNELS):
                win_data[channel_name] = all_windows[win_idx, :, ch_idx]  # 128 samples for this channel
            win_data['subject_id'] = f"UCI_S{subjects[win_idx]:02d}"
            win_data['activity_label'] = labels[win_idx]
            all_data.append(win_data)
        
        return pd.DataFrame(all_data)
    
    train_df = create_dataframe(train_signals, train_y, train_subjects, 'train')
    test_df = create_dataframe(test_signals, test_y, test_subjects, 'test')
    
    # Map activity IDs to names
    activity_map = dict(zip(activity_labels['id'], activity_labels['activity']))
    train_df['activity_label'] = train_df['activity_label'].map(activity_map)
    test_df['activity_label'] = test_df['activity_label'].map(activity_map)
    
    # Combine train and test
    uci_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Drop rows with missing sensor data
    uci_df = uci_df.dropna(subset=SENSOR_CHANNELS)
    
    print(f"UCI HAR loaded: {len(uci_df)} samples, {uci_df['subject_id'].nunique()} subjects")
    print(f"Activities: {dict(Counter(uci_df['activity_label']))}")
    
    return uci_df

def harmonize_labels(df, dataset_name):
    """Harmonize activity labels across datasets"""
    print(f"Harmonizing labels for {dataset_name}...")
    
    # Only clean KU-HAR activity names (remove number prefixes)
    if dataset_name == 'kuhar':
        df = df.copy()
        # Remove number prefixes like "0.Stand" -> "Stand"
        df['activity_label'] = df['activity_label'].str.replace(r'^\d+\.', '', regex=True)
        
        # Create mapping dictionary
        label_mapping = {}
        for shared_label, kuhar_activities in SHARED_ACTIVITIES.items():
            for kuhar_activity in kuhar_activities:
                label_mapping[kuhar_activity] = shared_label
        
        # Apply mapping
        before_harmonization = len(df)
        df['activity_label'] = df['activity_label'].map(label_mapping)
        
        # Drop rows with no mapping (activities not in shared space)
        df = df.dropna(subset=['activity_label'])
        after_harmonization = len(df)
        
        dropped = before_harmonization - after_harmonization
        if dropped > 0:
            print(f"Dropped {dropped:,} rows with activities not in shared label space")
    
    # UCI HAR labels are already in the correct format, no mapping needed
    elif dataset_name == 'ucihar':
        # Just ensure the labels match our expected format
        valid_activities = set(LABEL_MAP.keys())
        df = df[df['activity_label'].isin(valid_activities)]
    
    # Convert to numeric labels
    df['label'] = df['activity_label'].map(LABEL_MAP)
    
    print(f"After harmonization: {df['activity_label'].nunique()} activities")
    print(f"Activity distribution: {df['activity_label'].value_counts().to_dict()}")
    
    return df

def resample_to_target_hz(df, source_hz, target_hz):
    """
    Resample sensor data to match target sampling rate.
    KU-HAR is 100 Hz, UCI HAR is 50 Hz. We downsample KU-HAR to 50 Hz so that
    128-sample windows cover the same time duration (~2.56s) in both datasets.
    Without this, KU-HAR windows cover 1.28s while UCI covers 2.56s.
    """
    if source_hz == target_hz:
        return df
    
    ratio = source_hz / target_hz
    print(f"Resampling from {source_hz} Hz to {target_hz} Hz (ratio {ratio:.1f}:1)...")
    
    resampled_frames = []
    groups = df.groupby(['subject_id', 'activity_label'])
    
    for (subject, activity), group in groups:
        group = group.sort_index()
        # Take every nth sample (simple decimation)
        step = int(ratio)
        resampled = group.iloc[::step].copy()
        resampled_frames.append(resampled)
    
    result = pd.concat(resampled_frames, ignore_index=True)
    print(f"Resampled: {len(df)} → {len(result)} samples")
    return result

def create_windows(df, dataset_name):
    """Create sliding windows from time series data with better logging"""
    print(f"Creating windows for {dataset_name}...")
    
    windows = []
    window_labels = []
    window_subjects = []
    
    # Group by subject and activity
    groups = df.groupby(['subject_id', 'activity_label'])
    total_groups = len(groups)
    processed_groups = 0
    
    for (subject, activity), group in groups:
        processed_groups += 1
        if processed_groups % 50 == 0 or processed_groups == total_groups:
            print(f"  Processed {processed_groups}/{total_groups} groups")
        
        # Sort by time (assuming original order is temporal)
        group = group.sort_index()
        
        # Extract sensor data
        sensor_data = group[SENSOR_CHANNELS].values
        
        # Create sliding windows
        n_samples = len(sensor_data)
        if n_samples < WINDOW_SIZE:
            continue  # Skip groups that are too small
        
        for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            
            if end_idx <= n_samples:
                window = sensor_data[start_idx:end_idx]
                windows.append(window)
                window_labels.append(LABEL_MAP[activity])
                window_subjects.append(subject)
    
    if not windows:
        raise ValueError(f"No windows generated for {dataset_name}. Check that activity labels match label_map keys.")
    
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    window_subjects = np.array(window_subjects)
    
    print(f"Created {len(windows)} windows from {len(df)} samples")
    print(f"Window shape: {windows.shape}")
    
    # Print class distribution
    from collections import Counter
    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    counts = Counter(window_labels.tolist())
    print(f"Class distribution: { {inv_label_map[k]: v for k, v in sorted(counts.items())} }")
    
    return windows, window_labels, window_subjects

def subject_based_split(X, y, subjects, test_size=0.2, val_size=0.15):
    """Create train/val/test splits with subject separation and class coverage check"""
    print("Creating subject-based splits...")
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    n_classes = len(np.unique(y))
    
    # Try multiple random seeds to find a split where val covers all classes
    best_seed = RANDOM_SEED
    best_val_classes = 0
    
    for seed in range(RANDOM_SEED, RANDOM_SEED + 50):
        temp_subjects, test_subj = train_test_split(
            unique_subjects, test_size=test_size, random_state=seed
        )
        val_size_adjusted = val_size / (1 - test_size)
        train_subj, val_subj = train_test_split(
            temp_subjects, test_size=val_size_adjusted, random_state=seed
        )
        
        val_mask = np.isin(subjects, val_subj)
        val_classes = len(np.unique(y[val_mask]))
        
        if val_classes > best_val_classes:
            best_val_classes = val_classes
            best_seed = seed
        
        if val_classes == n_classes:
            break  # Found a split with all classes represented
    
    if best_val_classes < n_classes:
        print(f"WARNING: Best split covers {best_val_classes}/{n_classes} classes in val "
              f"(seed={best_seed}). Consider increasing val_size.")
    else:
        print(f"Found split with all {n_classes} classes in val (seed={best_seed})")
    
    # Use the best seed
    temp_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=best_seed
    )
    val_size_adjusted = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(
        temp_subjects, test_size=val_size_adjusted, random_state=best_seed
    )
    
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    
    # Create masks
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Train windows: {len(X_train)}")
    print(f"Val windows: {len(X_val)}")
    print(f"Test windows: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def normalize_data(X_train, X_val=None, X_test=None):
    """Z-score normalize using training set statistics"""
    print("Normalizing data...")
    
    # Calculate statistics from training set
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    
    # Avoid division by zero
    std = np.where(std < 1e-8, 1, std)
    
    # Normalize
    X_train_norm = (X_train - mean) / std
    
    normalized_sets = [X_train_norm]
    
    if X_val is not None:
        X_val_norm = (X_val - mean) / std
        normalized_sets.append(X_val_norm)
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        normalized_sets.append(X_test_norm)
    
    # Save normalization statistics
    scaler_stats = {
        'mean': mean.flatten().tolist(),
        'std': std.flatten().tolist()
    }
    
    print(f"Normalization complete. Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    return tuple(normalized_sets), scaler_stats

def process_kuhar():
    """Process KU-HAR dataset"""
    print("\n=== Processing KU-HAR Dataset ===")
    
    # Load and harmonize
    df = load_kuhar_data()
    df = harmonize_labels(df, 'kuhar')
    
    # Resample KU-HAR from 100 Hz to 50 Hz to match UCI HAR sampling rate.
    # This ensures 128-sample windows cover the same time duration (~2.56s)
    # in both datasets, which is critical for cross-dataset generalization.
    df = resample_to_target_hz(df, source_hz=100, target_hz=50)
    
    # Create windows
    X, y, subjects = create_windows(df, 'kuhar')
    
    # Split by subject
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = subject_based_split(X, y, subjects)
    
    # Normalize using training data statistics
    (X_train_norm, X_val_norm, X_test_norm), scaler_stats = normalize_data(X_train, X_val, X_test)

    # Save processed data (clean, no augmentation)
    KUHAR_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    np.save(KUHAR_PROCESSED_PATH / "X_train.npy", X_train_norm)
    np.save(KUHAR_PROCESSED_PATH / "y_train.npy", y_train)
    np.save(KUHAR_PROCESSED_PATH / "X_val.npy", X_val_norm)
    np.save(KUHAR_PROCESSED_PATH / "y_val.npy", y_val)
    np.save(KUHAR_PROCESSED_PATH / "X_test.npy", X_test_norm)
    np.save(KUHAR_PROCESSED_PATH / "y_test.npy", y_test)
    
    with open(KUHAR_PROCESSED_PATH / "scaler_stats.json", 'w') as f:
        json.dump(scaler_stats, f)
    
    print(f"KU-HAR processing complete. Saved to {KUHAR_PROCESSED_PATH}")
    print(f"Training samples: {len(X_train_norm)}")

def process_ucihar():
    """Process UCI HAR dataset"""
    print("\n=== Processing UCI HAR Dataset ===")
    
    # Load and harmonize
    df = load_ucihar_data()
    df = harmonize_labels(df, 'ucihar')
    
    # UCI HAR is already windowed, so extract windows directly
    print("Extracting pre-windowed data from UCI HAR...")
    
    windows = []
    window_labels = []
    window_subjects = []
    
    # Group by subject and activity
    groups = df.groupby(['subject_id', 'activity_label'])
    
    for (subject, activity), group in groups:
        # Each row in the group represents one window
        for _, row in group.iterrows():
            # Extract sensor data for this window (128 samples × 6 channels)
            window_data = np.array([
                row['acc_x'], row['acc_y'], row['acc_z'],
                row['gyro_x'], row['gyro_y'], row['gyro_z']
            ]).T  # Transpose to get (128, 6) shape
            
            windows.append(window_data)
            window_labels.append(LABEL_MAP[activity])
            window_subjects.append(subject)
    
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    window_subjects = np.array(window_subjects)
    
    print(f"Extracted {len(windows)} windows from UCI HAR")
    print(f"Window shape: {windows.shape}")
    
    # FIX: Split UCI into target-domain (unlabeled, for DANN alignment) and
    # held-out test (labeled, for cross-dataset evaluation) using subject separation.
    # Previously all UCI data was used as test only — this meant TCN_DA had no
    # target-domain samples to align against during training.
    unique_subjects = np.unique(window_subjects)
    train_subj, test_subj_list = train_test_split(
        unique_subjects, test_size=0.3, random_state=RANDOM_SEED
    )
    train_mask = np.isin(window_subjects, train_subj)
    test_mask  = np.isin(window_subjects, test_subj_list)

    X_target        = windows[train_mask]        # unlabeled target domain for DANN
    X_test          = windows[test_mask]
    y_test          = window_labels[test_mask]
    test_subjects   = window_subjects[test_mask]

    print(f"UCI target-domain (unlabeled, for DA training): {len(X_target)} windows")
    print(f"UCI test (labeled, for cross-dataset eval):     {len(X_test)} windows")

    # Normalize BOTH splits using KU-HAR training statistics.
    # Critical: using UCI own statistics would artificially reduce the domain gap
    # and make cross-dataset scores look better than real deployment would be.
    kuhar_stats_path = KUHAR_PROCESSED_PATH / "scaler_stats.json"
    if kuhar_stats_path.exists():
        with open(kuhar_stats_path, 'r') as f:
            scaler_stats = json.load(f)

        mean = np.array(scaler_stats['mean']).reshape(1, 1, -1)
        std  = np.array(scaler_stats['std']).reshape(1, 1, -1)
        std  = np.where(std < 1e-8, 1, std)

        X_target_norm = (X_target - mean) / std
        X_test_norm   = (X_test   - mean) / std
        print("Normalized UCI HAR (target + test) using KU-HAR training statistics")
    else:
        raise FileNotFoundError(
            f"KU-HAR scaler stats not found at {kuhar_stats_path}. "
            "Run process_kuhar() before process_ucihar() so that UCI data "
            "is normalized with KU-HAR training statistics."
        )

    # Save processed data
    UCIHAR_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # X_target_domain.npy: unlabeled UCI windows normalized with KU-HAR stats.
    # Used by TCN_DA during training for domain alignment (no labels needed).
    np.save(UCIHAR_PROCESSED_PATH / "X_target_domain.npy", X_target_norm)
    np.save(UCIHAR_PROCESSED_PATH / "X_test.npy",          X_test_norm)
    np.save(UCIHAR_PROCESSED_PATH / "y_test.npy",          y_test)
    np.save(UCIHAR_PROCESSED_PATH / "subjects.npy",        test_subjects)

    print(f"UCI HAR processing complete. Saved to {UCIHAR_PROCESSED_PATH}")

def create_combined_dataset():
    """Create the main handoff directory with all processed data"""
    print("\n=== Creating Combined Dataset ===")
    
    COMBINED_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    # Copy KU-HAR processed data
    kuhar_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
    for file in kuhar_files:
        src = KUHAR_PROCESSED_PATH / file
        dst = COMBINED_PROCESSED_PATH / file
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
    
    # Copy UCI HAR test data as cross-dataset test
    import shutil
    uci_src = UCIHAR_PROCESSED_PATH / "X_test.npy"
    uci_dst = COMBINED_PROCESSED_PATH / "X_cross_test.npy"
    if uci_src.exists():
        shutil.copy2(uci_src, uci_dst)
        shutil.copy2(UCIHAR_PROCESSED_PATH / "y_test.npy", COMBINED_PROCESSED_PATH / "y_cross_test.npy")

    # Copy UCI target-domain (unlabeled) data for TCN_DA DANN training
    target_src = UCIHAR_PROCESSED_PATH / "X_target_domain.npy"
    if target_src.exists():
        shutil.copy2(target_src, COMBINED_PROCESSED_PATH / "X_target_domain.npy")
        print("Copied X_target_domain.npy for TCN_DA domain alignment")
    else:
        print("WARNING: X_target_domain.npy not found — run process_ucihar() first")
    
    # Copy scaler stats and label map
    import shutil
    shutil.copy2(KUHAR_PROCESSED_PATH / "scaler_stats.json", COMBINED_PROCESSED_PATH)
    
    with open(COMBINED_PROCESSED_PATH / "label_map.json", 'w') as f:
        json.dump(LABEL_MAP, f)
    
    print(f"Combined dataset created at {COMBINED_PROCESSED_PATH}")
    
    # Print final dataset info
    print("\n=== Final Dataset Summary ===")
    for file in ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 
                 'X_test.npy', 'y_test.npy', 'X_cross_test.npy', 'y_cross_test.npy']:
        path = COMBINED_PROCESSED_PATH / file
        if path.exists():
            data = np.load(path)
            print(f"{file}: {data.shape}, dtype: {data.dtype}")
    
    # Create handoff verification
    handoff_info = {
        'description': 'Handoff package for Person B (deep learning)',
        'files_created': [
            'X_train.npy - KU-HAR training windows (N, 128, 6)',
            'y_train.npy - Training labels (N,)', 
            'X_val.npy - KU-HAR validation windows',
            'y_val.npy - Validation labels',
            'X_test.npy - KU-HAR test windows (within-dataset eval)',
            'y_test.npy - Test labels',
            'X_cross_test.npy - UCI HAR test windows (cross-dataset eval)',
            'X_target_domain.npy - UCI HAR unlabeled windows for TCN_DA DANN training (normalized with KU-HAR stats)',
            'y_cross_test.npy - Cross-dataset labels',
            'label_map.json - Activity to integer mapping',
            'scaler_stats.json - Normalization statistics from KU-HAR train'
        ],
        'input_conventions': {
            'shape': '(batch, 6, 128) - channels first for PyTorch',
            'normalization': 'Already z-score normalized using KU-HAR train stats',
            'labels': '0-5 integers corresponding to 6 UCI HAR activities'
        },
        'cross_dataset_note': 'UCI HAR data is normalized with KU-HAR statistics to simulate real deployment'
    }
    
    with open(COMBINED_PROCESSED_PATH / 'handoff_info.json', 'w') as f:
        json.dump(handoff_info, f, indent=2)
    
    print("\n✓ Handoff package ready for Person B")
    print(f"  All files saved to: {COMBINED_PROCESSED_PATH}")

def main():
    """Main preprocessing pipeline"""
    print("Starting preprocessing pipeline...")
    
    # Process both datasets
    process_kuhar()
    process_ucihar()
    
    # Create combined handoff dataset
    create_combined_dataset()
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess HAR datasets")
    parser.add_argument("--dataset", choices=["kuhar", "ucihar", "both"], default="both",
                       help="Which dataset(s) to process")
    
    args = parser.parse_args()
    
    if args.dataset == "kuhar":
        process_kuhar()
    elif args.dataset == "ucihar":
        process_ucihar()
    else:
        main()