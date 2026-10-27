
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, iirnotch

# Sampling rate for the Muse EEG device
muse_sampling = 256

# Create a filter to remove 60 Hz electrical noise
b_notch, a_notch = iirnotch(60, 30, fs=muse_sampling)

# Create a filter to keep frequencies between 4 and 30 Hz
b_band, a_band = butter(
    N=4, Wn=[4.0, 30.0], btype="bandpass", fs=muse_sampling
)


# Step 1: Clean the EEG signal
def preprocess_channel(signal):
    # Replace missing or infinite values with 0
    signal = np.nan_to_num(
        signal, nan=0, posinf=0, neginf=0
    )
    # Remove 60 Hz electrical noise
    filtered_signal = filtfilt(b_notch, a_notch, signal)
    # Keep frequencies between 4 and 30 Hz
    filtered_signal = filtfilt(b_band, a_band, filtered_signal)
    return filtered_signal


# Step 2: Calculate alpha-wave power
def calculate_alpha_power(signal):
    # Find the frequencies and power of the signal
    frequencies, power = welch(
        signal,
        fs=muse_sampling,
        nperseg=min(len(signal), 512)
    )
    # Start alpha power at 0
    alpha_power = 0
    # Go through each frequency
    for i in range(len(frequencies)):
        # Check if the frequency is between 8 and 12 Hz
        if frequencies[i] >= 8 and frequencies[i] <= 12:
            # Add the power at this frequency
            alpha_power = alpha_power + power[i]

    return alpha_power


# Step 3: Calculate valence and arousal
def compute_valence_arousal(df):
    # Get the left and right frontal EEG channels
    left_signal = df["AF7"].values
    right_signal = df["AF8"].values
    # Clean both signals
    left_clean = preprocess_channel(left_signal)
    right_clean = preprocess_channel(right_signal)

    # Calculate alpha power for each side
    left_alpha = calculate_alpha_power(left_clean)
    right_alpha = calculate_alpha_power(right_clean)

    # Calculate valence using frontal alpha asymmetry
    valence = (
        np.log(right_alpha + 1e-6)
        - np.log(left_alpha + 1e-6)
    )
    # Scale valence between -1 and 1
    valence = np.tanh(valence / 2)
    # Calculate the variance of each signal
    left_power = np.var(left_clean)
    right_power = np.var(right_clean)
    # Calculate arousal using total signal variance
    arousal = np.log(
        left_power + right_power + 1e-6
    )
    # Scale arousal between -1 and 1
    arousal = np.tanh(arousal / 10)
    return valence, arousal
# Step 4: Load the EEG data
df = pd.read_csv("Shriya_Elaine_raw_muse_session.csv")
# Step 5: Calculate valence and arousal
valence, arousal = compute_valence_arousal(df)
# Step 6: Print the results
print("Neurify preprocessing complete")
print("Valence:", round(valence, 4))
print("Arousal:", round(arousal, 4))
