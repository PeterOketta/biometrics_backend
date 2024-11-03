import numpy as np
from scipy import signal

def preprocess_ecg_series(ecg_data, sampling_rate=100):
    """Filter and normalize ECG signal"""
    # Convert to numpy array
    ecg_np = np.array(ecg_data, dtype=np.float32)
    
    # Design bandpass filter (0.5-40Hz)
    nyquist = sampling_rate / 2
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter
    filtered = signal.filtfilt(b, a, ecg_np)
    
    # Normalize
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    return normalized

def segment_signal(signal_data, window_size=200, max_segments=None):
    """
    Segment signal into non-overlapping windows of size 200
    Returns: list of numpy arrays, each of shape (200,)
    """
    # Convert to numpy array if not already
    signal_data = np.array(signal_data)
    
    # Calculate how many complete segments we can make
    total_samples = len(signal_data)
    n_complete_segments = total_samples // window_size
    
    if max_segments is not None:
        n_complete_segments = min(n_complete_segments, max_segments)
    
    if n_complete_segments == 0:
        raise ValueError("Signal too short to create any complete segments")
    
    # Create segments
    segments = []
    for i in range(n_complete_segments):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        segment = signal_data[start_idx:end_idx]
        segments.append(segment)
    
    return segments