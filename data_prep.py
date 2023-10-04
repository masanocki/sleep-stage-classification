import mne


# data_type="raw" - raw data from mne reader
# data_type="numb" - actual data in np.ndarray
def read_file(filepath, data_type="raw"):
    data = mne.io.read_raw_edf(filepath)
    if data_type == "numb":
        data = data.get_data()
    return data
