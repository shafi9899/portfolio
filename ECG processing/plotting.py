import matplotlib.pyplot as plt
import numpy as np

def visualize_ecg_raw_vs_filtered(raw_signal, filtered_signal, fs=360):
    time = np.arange(len(raw_signal)) / fs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, raw_signal, label="Raw ECG Signal")
    plt.title("Raw ECG Signal")
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_signal, label="Filtered ECG Signal", color="green")
    plt.title("Filtered ECG Signal")
    plt.tight_layout()
    plt.show()

def visualize_ecg_with_r_peaks(ecg_signal, peaks, fs=360):
    time = np.arange(len(ecg_signal)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(time, ecg_signal)
    plt.plot(peaks / fs, ecg_signal[peaks], 'rx')
    plt.title("ECG with R-Peaks")
    plt.show()

def visualize_heart_rate(heart_rate):
    plt.figure()
    plt.plot(heart_rate)
    plt.title("Heart Rate Over Time")
    plt.show()
