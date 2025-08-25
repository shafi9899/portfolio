import pandas as pd
from bandpass_filter import bandpass_filter

def load_and_filter_ecg_signal(csv_file, lowcut=0.5, highcut=50, sampling_rate=360):
    """
    Load the ECG signal from a CSV file and apply the bandpass filter.

    Parameters:
    csv_file (str): Path to the CSV file containing ECG data.
    lowcut (float): Lower cutoff frequency for the bandpass filter.
    highcut (float): Upper cutoff frequency for the bandpass filter.
    sampling_rate (int): Sampling frequency of the ECG signal.

    Returns:
    array: Filtered ECG signal.
    """
    ecg_data = pd.read_csv(csv_file)
    print("Columns in the ECG data:", ecg_data.columns)  # Print columns for debugging
    ecg_data.columns = ecg_data.columns.str.strip().str.replace("''", "").str.replace("'", "")
     
    if 'MLII' not in ecg_data.columns:
        raise ValueError("ECG signal column 'MLII' is missing in the data.")
    
    raw_signal = ecg_data['MLII'].values  # Accessing the MLII signal (if it exists)
    
    # Apply bandpass filter
    filtered_signal = bandpass_filter(raw_signal, lowcut, highcut, sampling_rate)
    
    return raw_signal, filtered_signal  # Return both raw and filtered signals
