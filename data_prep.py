import mne
import numpy as np
from mne.time_frequency import psd_array_welch


# data_type is for selecting how do you want to load data:
# df - pandas dataframe
# numarr - numpy array
# raw - raw data
# drop is for droping ecg and eog channels leaving only eeg
def read_file(filepath, data_type="raw", drop=False):
    if data_type not in ["raw", "df", "numarr"]:
        return "Enter correct data type: [raw, numarr, df]"
    data = mne.io.read_raw_edf(filepath, infer_types=True, preload=True)
    if drop:
        data.drop_channels(["chin", "E1-M2", "E2-M2"])
    if data_type == "numarr":
        data = data.get_data()
    if data_type == "df":
        data = data.to_data_frame()
    return data


# for this step data have to be raw
def preprocess(data):
    epochs = mne.make_fixed_length_epochs(data, duration=30, preload=True)
    return epochs


def eeg_power_band(epochs):
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30],
    }
    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)
