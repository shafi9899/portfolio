import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_and_filter_ecg import load_and_filter_ecg_signal
from r_peak_detection import detect_r_peaks
from rr_interval_heart_rate import calculate_rr_intervals, calculate_heart_rate
from plotting import visualize_ecg_with_r_peaks, visualize_heart_rate

def visualize_signals_with_zoom(raw_signal, filtered_signal, peaks):
    """
    Visualizes the raw ECG signal, filtered ECG signal, and R-peaks with zoomed-in views.

    Parameters:
        raw_signal (array-like): The raw ECG signal.
        filtered_signal (array-like): The filtered ECG signal.
        peaks (array-like): Indices of detected R-peaks.
    """
    zoom_start = 1000  # Start index for zoom
    zoom_end = 2000    # End index for zoom

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle("ECG Signal Processing - Step-by-Step (With Zoom)", fontsize=16)

    # 1. Raw Signal
    axes[0, 0].plot(raw_signal, color="blue", lw=1)
    axes[0, 0].set_title("Raw ECG Signal (Full)")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid()

    axes[0, 1].plot(range(zoom_start, zoom_end), raw_signal[zoom_start:zoom_end], color="blue", lw=1)
    axes[0, 1].set_title("Raw ECG Signal (Zoomed-In)")
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid()

    # 2. Filtered Signal
    axes[1, 0].plot(filtered_signal, color="green", lw=1)
    axes[1, 0].set_title("Filtered ECG Signal (Full)")
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid()

    axes[1, 1].plot(range(zoom_start, zoom_end), filtered_signal[zoom_start:zoom_end], color="green", lw=1)
    axes[1, 1].set_title("Filtered ECG Signal (Zoomed-In)")
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid()

    # 3. Filtered Signal with R-peaks
    axes[2, 0].plot(filtered_signal, color="orange", lw=1)
    axes[2, 0].scatter(peaks, filtered_signal[peaks], color="red", label="R-peaks")
    axes[2, 0].set_title("Filtered Signal with R-peaks (Full)")
    axes[2, 0].set_xlabel("Sample Index")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].legend()
    axes[2, 0].grid()

    axes[2, 1].plot(range(zoom_start, zoom_end), filtered_signal[zoom_start:zoom_end], color="orange", lw=1)
    zoomed_peaks = [p for p in peaks if zoom_start <= p < zoom_end]
    axes[2, 1].scatter(zoomed_peaks, filtered_signal[zoomed_peaks], color="red", label="R-peaks")
    axes[2, 1].set_title("Filtered Signal with R-peaks (Zoomed-In)")
    axes[2, 1].set_xlabel("Sample Index")
    axes[2, 1].set_ylabel("Amplitude")
    axes[2, 1].legend()
    axes[2, 1].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.show()

def main():
    csv_file = r"D:/5th Semester\Signals and systems\DATABASE\Compressed\ECG FILE\mitbih_database\100.csv"

    try:
        print("Step 1: Loading and filtering the ECG signal...")
        raw_signal, filtered_signal = load_and_filter_ecg_signal(csv_file)
        print("-> ECG signal loaded and filtered successfully.")

        print("Step 2: Detecting R-peaks in the ECG signal...")
        peaks = detect_r_peaks(filtered_signal)
        print(f"-> R-peaks detected at indices: {peaks[:5]} (showing first 5). Total peaks: {len(peaks)}")

        print("Step 3: Visualizing the ECG signal processing steps with zoomed-in views...")
        visualize_signals_with_zoom(raw_signal, filtered_signal, peaks)
        print("-> Visualization completed.")

        print("Step 4: Calculating RR intervals...")
        rr_intervals = calculate_rr_intervals(peaks)
        print(f"-> First 5 RR intervals: {rr_intervals[:5]}")

        print("Step 5: Calculating heart rate...")
        heart_rate = calculate_heart_rate(rr_intervals)
        print(f"-> First 5 heart rates (BPM): {heart_rate[:5]}")

        print("Step 6: Visualizing heart rate trend over time...")
        visualize_heart_rate(heart_rate)
        print("-> Heart rate visualization completed.")

        print("Step 7: Real-time heart rate simulation...")
        plt.ion()  # Enable interactive mode for real-time plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Real-Time Heart Rate Simulation")
        ax.set_xlim(0, 10)  # Initial time window
        ax.set_ylim(40, 180)  # Typical heart rate range in BPM
        ax.set_xlabel("Time (Index)")
        ax.set_ylabel("Heart Rate (BPM)")

        line, = ax.plot([], [], color="red", lw=2)
        x_data, y_data = [], []
        bpm_text = ax.text(0.05, 0.95, "Heart Rate: 0 BPM", transform=ax.transAxes, fontsize=14,
                           verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))

        max_updates = 100
        update_interval = 5

        for i, hr in enumerate(heart_rate):
            if i >= max_updates:
                print("-> Stopping real-time simulation after reaching the maximum updates.")
                break

            x_data.append(i)
            y_data.append(hr)
            line.set_data(x_data, y_data)
            ax.set_xlim(max(0, i - 10), i)

            if i % update_interval == 0:
                bpm_text.set_text(f"Heart Rate: {hr} BPM")

            plt.draw()
            plt.pause(0.1)

        plt.ioff()  # Turn off interactive mode
        plt.show()
        print("-> Real-time heart rate simulation completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
