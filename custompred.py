import numpy as np
import mne
import matplotlib.pyplot as plt
import data_prep as dp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score

data = dp.read_file("datasets/recordings/SN001.edf", data_type="raw", drop=False)

annot_train = mne.read_annotations("datasets/recordings/SN001_sleepscoring.edf")
data.set_annotations(annot_train)

# data.plot(
#     start=60,
#     duration=60,
#     scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1),
# )

event_id = {
    "Sleep stage W": 1,
    "Sleep stage N1": 2,
    "Sleep stage N2": 3,
    "Sleep stage N3": 4,
    "Sleep stage R": 5,
}

# annot_train.crop(annot_train[1]["onset"] - 30 * 60, annot_train[-2]["onset"] + 30 * 60)
# data.set_annotations(annot_train)

events_train, _ = mne.events_from_annotations(
    data, event_id=event_id, chunk_duration=30.0
)

# fig = mne.viz.plot_events(
#     events_train,
#     event_id=event_id,
#     sfreq=data.info["sfreq"],
#     first_samp=events_train[0, 0],
# )

stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

tmax = 30.0 - 1.0 / data.info["sfreq"]

epochs_train = mne.Epochs(
    raw=data, events=events_train, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None
)


data_test = dp.read_file("datasets/recordings/SN002.edf", data_type="raw", drop=False)
annot_test = mne.read_annotations("datasets/recordings/SN002_sleepscoring.edf")
data_test.set_annotations(annot_test)
events_test, _ = mne.events_from_annotations(
    data_test, event_id=event_id, chunk_duration=30.0
)
epochs_test = mne.Epochs(
    raw=data_test,
    events=events_test,
    event_id=event_id,
    tmin=0.0,
    tmax=tmax,
    baseline=None,
)

pipe = make_pipeline(
    FunctionTransformer(dp.eeg_power_band, validate=False),
    RandomForestClassifier(n_estimators=100, random_state=42),
)

y_train = epochs_train.events[:, 2]
pipe.fit(epochs_train, y_train)

y_pred = pipe.predict(epochs_test)

y_test = epochs_test.events[:, 2]
acc = accuracy_score(y_test, y_pred)
print(f"acc: {acc}")

# plt.show()
