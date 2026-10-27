import numpy as np 
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch 

df = pd.read_csv("Shriya_Elaine_raw_muse_session.csv")

#print(df)

af7 = df["AF7"].values
af8 = df["AF8"].values

af7 = np.nan_to_num(af7)
af8 = np.nan_to_num(af8)

# print(af7)
# print(af8)

muse_sampling = 256 
# Filter to reduce 60 Hz electrical noise
b_notch, a_notch = iirnotch(60.0, 30.0, fs=muse_sampling)

# apply the notch filter to both EEG channels
af7_notched = filtfilt(b_notch, a_notch, af7)
af8_notched = filtfilt(b_notch, a_notch, af8)

# print(af7_notched[:5])
# print(af8_notched[:5])

# Create filter that keeps frequencies between 4 and 30 Hz
b_band, a_band = butter(4, [4.0, 30.0], btype="bandpass", fs=muse_sampling)

af7_clean = filtfilt(b_band, a_band, af7_notched)
af8_clean = filtfilt(b_band, a_band, af8_notched)

#print(af7_clean[:5])
#print(af8_clean[:5])


def get_alpha_power(signal):
    # Find the frequencies and their power
    freqs, psd = welch(signal, fs=muse_sampling, nperseg=min(len(signal), muse_sampling * 2))
    # Find which frequencies are between 8 and 12 Hz
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    # Select the power values at those frequencies and add them
    alpha_power = np.sum(psd[alpha_mask])
    return alpha_power


# Calculate alpha power for each EEG channel
alpha_af7 = get_alpha_power(af7_clean)
alpha_af8 = get_alpha_power(af8_clean)

#print("AF7 alpha power:", alpha_af7)
#print("AF8 alpha power:", alpha_af8)


def get_beta_power(signal):
    # Find the frequencies and their power
    freqs, psd = welch(
        signal,
        fs=muse_sampling,
        nperseg=min(len(signal), muse_sampling * 2)
    )

    # Find frequencies between 13 and 30 Hz
    beta_mask = (freqs >= 13) & (freqs <= 30)

    # Add the power values in that frequency range
    beta_power = np.sum(psd[beta_mask])

    return beta_power


beta_af7 = get_beta_power(af7_clean)
beta_af8 = get_beta_power(af8_clean)

#print("AF7 beta power:", beta_af7)
#print("AF8 beta power:", beta_af8)


def get_theta_power(signal):
    # Find the frequencies and their power
    freqs, psd = welch(
        signal,
        fs=muse_sampling,
        nperseg=min(len(signal), muse_sampling * 2)
    )

    # Find frequencies between 4 and 8 Hz
    theta_mask = (freqs >= 4) & (freqs < 8)

    # Add the power values in that frequency range
    theta_power = np.sum(psd[theta_mask])

    return theta_power


theta_af7 = get_theta_power(af7_clean)
theta_af8 = get_theta_power(af8_clean)

#print("AF7 theta power:", theta_af7)
#print("AF8 theta power:", theta_af8)

theta_af7 = get_theta_power(af7_clean)
theta_af8 = get_theta_power(af8_clean)

#print("AF7 theta power:", theta_af7)
#print("AF8 theta power:", theta_af8)


# Calculate valence using alpha power differences
raw_valence = np.log(alpha_af8 + 1e-6) - np.log(alpha_af7 + 1e-6)

valence = np.tanh(raw_valence / 2.0)


# Calculate arousal using the cleaned EEG signals
total_power_af7 = np.var(af7_clean)
total_power_af8 = np.var(af8_clean)

raw_arousal = np.log(total_power_af7 + total_power_af8 + 1e-6)

arousal = np.tanh(raw_arousal / 10.0)

print("--- Preprocessing Complete ---")
print(f"Valence: {valence:.4f}")
print(f"Arousal: {arousal:.4f}")
