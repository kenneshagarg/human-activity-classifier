"""
Feature extraction module for Human Activity Recognition
Extracts 96 hand-crafted features from sensor windows
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from config import *

class FeatureExtractor:
    """Extract hand-crafted features from sensor windows"""
    
    def __init__(self):
        self.feature_names = []
        self._setup_feature_names()
    
    def _setup_feature_names(self):
        """Setup feature names for all 176 features.

        Order must match extract_features() exactly:
          Time  : for axis in (x,y,z): acc_axis(16), gyro_axis(16)  → 96 total
          Freq  : for axis in (x,y,z): acc_axis(8),  gyro_axis(8)   → 48 total
          Stat  : acc(16), gyro(16)                                  → 32 total
        """
        time_features = ['mean', 'std', 'var', 'min', 'max', 'range', 'median', 'mad',
                         'skew', 'kurtosis', 'energy', 'rms', 'peak_to_peak', 'iqr',
                         'zero_crossing_rate', 'mean_abs_diff']

        freq_features = ['dominant_freq', 'freq_energy', 'freq_entropy', 'spectral_centroid',
                         'spectral_rolloff', 'spectral_bandwidth', 'spectral_flux', 'peak_count']

        statistical_features = ['correlation_xy', 'correlation_xz', 'correlation_yz',
                                 'signal_magnitude_area', 'mean_abs_jerk', 'entropy', 'autocorr_max',
                                 'linearity_x', 'linearity_y', 'linearity_z',
                                 'std_x', 'std_y', 'std_z',
                                 'mean_abs_deviation_x', 'mean_abs_deviation_y', 'mean_abs_deviation_z']

        # Time features — interleaved: acc_x, gyro_x, acc_y, gyro_y, acc_z, gyro_z
        for axis in ['x', 'y', 'z']:
            for sensor in ['acc', 'gyro']:
                for feat in time_features:
                    self.feature_names.append(f"{sensor}_{axis}_{feat}")

        # Frequency features — same interleaving
        for axis in ['x', 'y', 'z']:
            for sensor in ['acc', 'gyro']:
                for feat in freq_features:
                    self.feature_names.append(f"{sensor}_{axis}_{feat}")

        # Statistical features — acc block then gyro block
        for sensor in ['acc', 'gyro']:
            for feat in statistical_features:
                self.feature_names.append(f"{sensor}_{feat}")

        assert len(self.feature_names) == N_FEATURES, \
            f"Expected {N_FEATURES} features, got {len(self.feature_names)}"
    
    def extract_time_features(self, signal):
        """Extract 16 time-domain features from a signal"""
        features = []
        
        # Basic statistics
        features.append(np.mean(signal))                    # mean
        features.append(np.std(signal))                     # std
        features.append(np.var(signal))                     # var
        features.append(np.min(signal))                     # min
        features.append(np.max(signal))                     # max
        features.append(np.max(signal) - np.min(signal))    # range
        features.append(np.median(signal))                  # median
        features.append(np.median(np.abs(signal - np.median(signal))))  # mad
        
        # Higher order statistics
        features.append(stats.skew(signal))                 # skew
        features.append(stats.kurtosis(signal))             # kurtosis
        
        # Energy and power
        features.append(np.sum(signal ** 2))               # energy
        features.append(np.sqrt(np.mean(signal ** 2)))      # rms
        features.append(np.max(signal) - np.min(signal))   # peak_to_peak
        features.append(np.percentile(signal, 75) - np.percentile(signal, 25))  # iqr
        
        # Signal characteristics
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features.append(len(zero_crossings) / len(signal))  # zero_crossing_rate
        features.append(np.mean(np.abs(np.diff(signal))))    # mean_abs_diff
        
        return np.array(features)
    
    def extract_frequency_features(self, signal):
        """Extract 8 frequency-domain features from a signal"""
        # Compute FFT
        fft_vals = fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/SAMPLING_RATE)
        
        # Take only positive frequencies
        pos_mask = fft_freq > 0
        fft_vals = fft_vals[pos_mask]
        fft_freq = fft_freq[pos_mask]
        fft_power = np.abs(fft_vals) ** 2
        
        features = []
        
        # Dominant frequency
        if len(fft_power) > 0:
            dominant_freq_idx = np.argmax(fft_power)
            features.append(fft_freq[dominant_freq_idx])    # dominant_freq
        else:
            features.append(0.0)
        
        # Frequency energy
        features.append(np.sum(fft_power))                 # freq_energy
        
        # Frequency entropy
        if np.sum(fft_power) > 0:
            power_norm = fft_power / np.sum(fft_power)
            power_norm = power_norm[power_norm > 0]
            freq_entropy = -np.sum(power_norm * np.log2(power_norm))
            features.append(freq_entropy)                  # freq_entropy
        else:
            features.append(0.0)
        
        # Spectral features
        if len(fft_power) > 0 and np.sum(fft_power) > 0:
            power_norm = fft_power / np.sum(fft_power)
            
            # Spectral centroid
            spectral_centroid = np.sum(fft_freq * power_norm)
            features.append(spectral_centroid)              # spectral_centroid
            
            # Spectral rolloff (85% of energy)
            cumsum_power = np.cumsum(power_norm)
            rolloff_idx = np.where(cumsum_power >= 0.85)[0]
            if len(rolloff_idx) > 0:
                features.append(fft_freq[rolloff_idx[0]])   # spectral_rolloff
            else:
                features.append(fft_freq[-1])
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(power_norm * (fft_freq - spectral_centroid) ** 2))
            features.append(spectral_bandwidth)            # spectral_bandwidth
            
            # Spectral flux (difference with previous window - simplified)
            features.append(np.std(fft_power))             # spectral_flux
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Peak count in frequency domain
        peaks, _ = find_peaks(np.abs(fft_vals), height=np.max(np.abs(fft_vals)) * 0.1)
        features.append(len(peaks))                        # peak_count
        
        return np.array(features)
    
    def extract_statistical_features(self, signals_3d):
        """Extract 15 statistical features from 3-axis sensor data (optimized for speed)"""
        # signals_3d shape: (window_size, 3)
        x, y, z = signals_3d[:, 0], signals_3d[:, 1], signals_3d[:, 2]
        
        features = []
        
        # Correlations between axes
        features.append(np.corrcoef(x, y)[0, 1] if len(set(x)) > 1 else 0)  # correlation_xy
        features.append(np.corrcoef(x, z)[0, 1] if len(set(x)) > 1 else 0)  # correlation_xz
        features.append(np.corrcoef(y, z)[0, 1] if len(set(y)) > 1 else 0)  # correlation_yz
        
        # Signal Magnitude Area (SMA)
        sma = np.sum((np.abs(x) + np.abs(y) + np.abs(z)) / len(x))
        features.append(sma)                              # signal_magnitude_area
        
        # Mean absolute jerk (rate of change of acceleration)
        jerk_x = np.diff(x)
        jerk_y = np.diff(y)
        jerk_z = np.diff(z)
        mean_abs_jerk = np.mean(np.abs(jerk_x) + np.abs(jerk_y) + np.abs(jerk_z))
        features.append(mean_abs_jerk)                     # mean_abs_jerk
        
        # Simple entropy (much faster than approximate/sample entropy)
        def simple_entropy(signal):
            hist, _ = np.histogram(signal, bins=16, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        
        entropy = simple_entropy(x) + simple_entropy(y) + simple_entropy(z)
        features.append(entropy)                          # entropy
        
        # Maximum autocorrelation (simplified)
        def simple_autocorr(signal):
            # Use only first 20 lags instead of full autocorrelation
            n = len(signal)
            signal = signal - np.mean(signal)
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[n-1:n+20] / autocorr[n-1]
            return np.max(np.abs(autocorr[1:]))  # Skip lag 0
        
        max_autocorr = max(simple_autocorr(x), simple_autocorr(y), simple_autocorr(z))
        features.append(max_autocorr)                      # autocorr_max
        
        # Linearity (fit quality of linear trend)
        time_points = np.arange(len(x))
        
        # Linear fits
        x_fit = np.polyfit(time_points, x, 1)
        y_fit = np.polyfit(time_points, y, 1)
        z_fit = np.polyfit(time_points, z, 1)
        
        # R-squared values
        x_pred = np.polyval(x_fit, time_points)
        y_pred = np.polyval(y_fit, time_points)
        z_pred = np.polyval(z_fit, time_points)
        
        x_r2 = 1 - np.sum((x - x_pred)**2) / np.sum((x - np.mean(x))**2)
        y_r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        z_r2 = 1 - np.sum((z - z_pred)**2) / np.sum((z - np.mean(z))**2)
        
        features.append(x_r2)                              # linearity_x
        features.append(y_r2)                              # linearity_y
        features.append(z_r2)                              # linearity_z
        
        # Additional simple statistical features
        features.append(np.std(x))                          # std_x
        features.append(np.std(y))                          # std_y
        features.append(np.std(z))                          # std_z
        features.append(np.mean(np.abs(x - np.mean(x))))    # mean_abs_deviation_x
        features.append(np.mean(np.abs(y - np.mean(y))))    # mean_abs_deviation_y
        features.append(np.mean(np.abs(z - np.mean(z))))    # mean_abs_deviation_z
        
        return features
    
    def extract_features(self, window):
        """Extract all 96 features from a single window
        
        Args:
            window: numpy array of shape (window_size, 6) with [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            features: numpy array of shape (96,)
        """
        features = []
        
        # Extract accelerometer and gyroscope data
        acc_data = window[:, :3]  # acc_x, acc_y, acc_z
        gyro_data = window[:, 3:6]  # gyro_x, gyro_y, gyro_z
        
        # Time domain features (16 per axis = 48 total)
        for i in range(3):  # x, y, z axes
            # Accelerometer
            features.extend(self.extract_time_features(acc_data[:, i]))
            # Gyroscope  
            features.extend(self.extract_time_features(gyro_data[:, i]))
        
        # Frequency domain features (8 per axis = 24 total)
        for i in range(3):  # x, y, z axes
            # Accelerometer
            features.extend(self.extract_frequency_features(acc_data[:, i]))
            # Gyroscope
            features.extend(self.extract_frequency_features(gyro_data[:, i]))
        
        # Statistical features (24 per sensor type = 48 total)
        features.extend(self.extract_statistical_features(acc_data))
        features.extend(self.extract_statistical_features(gyro_data))
        
        features = np.array(features)
        
        # Handle any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        assert len(features) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(features)}"
        
        return features
    
    def extract_features_batch(self, windows):
        """Extract features from multiple windows
        
        Args:
            windows: numpy array of shape (n_windows, window_size, 6)
        
        Returns:
            features: numpy array of shape (n_windows, 96)
        """
        print(f"Extracting features from {len(windows)} windows...")
        
        features_list = []
        for i, window in enumerate(windows):
            # More frequent progress updates for small datasets
            if len(windows) < 1000:
                if i % 100 == 0 or i == len(windows) - 1:
                    print(f"Processed {i+1}/{len(windows)} windows ({(i+1)/len(windows)*100:.1f}%)")
            else:
                if i % 1000 == 0 or i == len(windows) - 1:
                    print(f"Processed {i+1}/{len(windows)} windows ({(i+1)/len(windows)*100:.1f}%)")
            
            features = self.extract_features(window)
            features_list.append(features)
        
        features_array = np.array(features_list)
        print(f"Feature extraction complete: {features_array.shape}")
        
        return features_array
    
    def get_feature_names(self):
        """Get the names of all features"""
        return self.feature_names.copy()
    
    def get_feature_group_indices(self, group_name):
        """Get indices of features belonging to a specific group"""
        if group_name not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group_name}")
        return FEATURE_GROUPS[group_name]

def extract_features_for_classical_ml():
    """Extract features for classical ML models"""
    print("Extracting features for classical ML...")
    
    # Load processed data
    X_train = np.load(COMBINED_PROCESSED_PATH / "X_train.npy")
    y_train = np.load(COMBINED_PROCESSED_PATH / "y_train.npy")
    X_val = np.load(COMBINED_PROCESSED_PATH / "X_val.npy")
    y_val = np.load(COMBINED_PROCESSED_PATH / "y_val.npy")
    X_test = np.load(COMBINED_PROCESSED_PATH / "X_test.npy")
    y_test = np.load(COMBINED_PROCESSED_PATH / "y_test.npy")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    X_train_features = extractor.extract_features_batch(X_train)
    X_val_features = extractor.extract_features_batch(X_val)
    X_test_features = extractor.extract_features_batch(X_test)
    
    # Save features
    np.save(COMBINED_PROCESSED_PATH / "X_train_features.npy", X_train_features)
    np.save(COMBINED_PROCESSED_PATH / "X_val_features.npy", X_val_features)
    np.save(COMBINED_PROCESSED_PATH / "X_test_features.npy", X_test_features)
    
    # Save feature names
    feature_names = extractor.get_feature_names()
    with open(COMBINED_PROCESSED_PATH / "feature_names.json", 'w') as f:
        import json
        json.dump(feature_names, f)
    
    print(f"Features saved to {COMBINED_PROCESSED_PATH}")
    print(f"Feature shape: {X_train_features.shape}")
    
    return X_train_features, X_val_features, X_test_features

if __name__ == "__main__":
    extract_features_for_classical_ml()
