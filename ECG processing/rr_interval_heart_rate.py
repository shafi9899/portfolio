import numpy as np

def calculate_rr_intervals(r_peaks, sampling_rate=360):
    """
    Calculate and validate RR intervals from detected R-peaks.
    
    Parameters:
    - r_peaks (array): Indices of R-peaks in the ECG signal.
    - sampling_rate (int): Sampling frequency (default = 360 Hz).

    Returns:
    - rr_intervals (array): Validated RR intervals in seconds.
    """
    rr_intervals_samples = np.diff(r_peaks)
    rr_intervals_seconds = rr_intervals_samples / sampling_rate
    rr_intervals_valid = rr_intervals_seconds[(rr_intervals_seconds > 0.3) & (rr_intervals_seconds < 2.0)]

    print(f"Valid RR intervals: {rr_intervals_valid}")
    return rr_intervals_valid

def calculate_heart_rate(rr_intervals):
    """
    Calculate and validate heart rate from RR intervals.
    
    Parameters:
    - rr_intervals (array): Validated RR intervals in seconds.

    Returns:
    - heart_rate (array): Validated heart rate in beats per minute (bpm).
    """
    heart_rate = 60 / rr_intervals
    heart_rate_valid = heart_rate[(heart_rate > 40) & (heart_rate < 200)]

    print(f"Valid Heart Rates: {heart_rate_valid}")
    return heart_rate_valid
