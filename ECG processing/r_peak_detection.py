from scipy.signal import find_peaks
import numpy as np

def detect_r_peaks(ecg_signal, height=0.5, distance=100, threshold_factor=1.5, fs=360):
    """
    Detect R-peaks in the ECG signal using the find_peaks method with enhanced thresholding and RR interval validation.
    
    Parameters:
    - ecg_signal (array): The filtered ECG signal.
    - height (float): Minimum height to detect peaks (default = 0.5).
    - distance (int): Minimum distance between peaks (default = 100 samples).
    - threshold_factor (float): Factor to adjust dynamic threshold (default = 1.5).
    - fs (int): Sampling frequency (default = 360 Hz).
    
    Returns:
    - peaks (array): Indices of the detected R-peaks.
    """
    mean_signal = np.mean(ecg_signal)
    std_signal = np.std(ecg_signal)
    adaptive_threshold = mean_signal + threshold_factor * std_signal

    peaks, _ = find_peaks(ecg_signal, height=adaptive_threshold, distance=distance)

    min_rr_interval = int(0.3 * fs)  
    valid_peaks = []

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] > min_rr_interval:  
            valid_peaks.append(peaks[i])
    valid_peaks = [peaks[0]] + valid_peaks  # Always keep the first 

    return np.array(valid_peaks)
