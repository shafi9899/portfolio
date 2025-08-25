from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    """
    Apply a bandpass filter to the ECG signal to remove noise.
    
    Parameters:
    - signal (array): Raw ECG signal.
    - lowcut (float): Lower cutoff frequency (Hz).
    - highcut (float): Upper cutoff frequency (Hz).
    - fs (int): Sampling frequency (Hz).
    - order (int): Filter order.

    Returns:
    - array: Filtered ECG signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
