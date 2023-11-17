import numpy as np
import mne
import matplotlib.pyplot as plt
import data_prep as dp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score


class CustomTrainPredict:
    def __init__(
        self,
        train_raw_data_path,
        train_annotations_path,
        test_raw_data_path,
        test_annotations_path,
        n_estimators,
        min_samples_leaf,
        max_features,
        random_state,
    ):
        self.train_raw_data_path_ = train_raw_data_path
        self.train_annotations_path_ = train_annotations_path
        self.test_raw_data_path_ = test_raw_data_path
        self.test_annotations_path_ = test_annotations_path
        self.n_estimators_ = n_estimators
        self.min_samples_leaf_ = min_samples_leaf
        self.max_features_ = max_features
        self.random_state_ = random_state

    def predict(self):
        train_data = dp.read_file(
            self.train_raw_data_path_, data_type="raw", drop=False
        )

        annot_train = mne.read_annotations(self.train_annotations_path_)
        train_data.set_annotations(annot_train)

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

        events_train, _ = mne.events_from_annotations(
            train_data, event_id=event_id, chunk_duration=30.0
        )

        # fig = mne.viz.plot_events(
        #     events_train,
        #     event_id=event_id,
        #     sfreq=data.info["sfreq"],
        #     first_samp=events_train[0, 0],
        # )

        # stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        tmax = 30.0 - 1.0 / train_data.info["sfreq"]

        epochs_train = mne.Epochs(
            raw=train_data,
            events=events_train,
            event_id=event_id,
            tmin=0.0,
            tmax=tmax,
            baseline=None,
        )

        test_data = dp.read_file(self.test_raw_data_path_, data_type="raw", drop=False)
        annot_test = mne.read_annotations(self.test_annotations_path_)
        test_data.set_annotations(annot_test)
        events_test, _ = mne.events_from_annotations(
            test_data, event_id=event_id, chunk_duration=30.0
        )
        epochs_test = mne.Epochs(
            raw=test_data,
            events=events_test,
            event_id=event_id,
            tmin=0.0,
            tmax=tmax,
            baseline=None,
        )

        pipe = make_pipeline(
            FunctionTransformer(dp.eeg_power_band, validate=False),
            RandomForestClassifier(
                n_estimators=self.n_estimators_,
                min_samples_leaf=self.min_samples_leaf_,
                max_features=self.max_features_,
                random_state=self.random_state_,
            ),
        )

        y_train = epochs_train.events[:, 2]
        pipe.fit(epochs_train, y_train)

        y_pred = pipe.predict(epochs_test)

        y_test = epochs_test.events[:, 2]
        acc = accuracy_score(y_test, y_pred)
        print(f"acc: {acc}")

        # plt.show()
