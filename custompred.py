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
        self.train_data = dp.read_file(
            self.train_raw_data_path_, data_type="raw", drop=True
        )

        self.annot_train = mne.read_annotations(self.train_annotations_path_)

        self.train_data.set_annotations(self.annot_train)

        # # data.plot(
        # #     start=60,
        # #     duration=60,
        # #     scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1),
        # # )

        self.event_id = {
            "Sleep stage W": 1,
            "Sleep stage N1": 2,
            "Sleep stage N2": 3,
            "Sleep stage N3": 4,
            "Sleep stage R": 5,
        }

        self.events_train, _ = mne.events_from_annotations(
            self.train_data, event_id=self.event_id, chunk_duration=30.0
        )

        # # fig = mne.viz.plot_events(
        # #     events_train,
        # #     event_id=event_id,
        # #     sfreq=data.info["sfreq"],
        # #     first_samp=events_train[0, 0],
        # # )

        # # stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        self.tmax = 30.0 - 1.0 / self.train_data.info["sfreq"]

        self.epochs_train = mne.Epochs(
            raw=self.train_data,
            events=self.events_train,
            event_id=self.event_id,
            tmin=0.0,
            tmax=self.tmax,
            baseline=None,
        )

        self.test_data = dp.read_file(
            self.test_raw_data_path_, data_type="raw", drop=True
        )

        self.annot_test = mne.read_annotations(self.test_annotations_path_)
        self.test_data.set_annotations(self.annot_test)
        self.events_test, _ = mne.events_from_annotations(
            self.test_data, event_id=self.event_id, chunk_duration=30.0
        )
        self.epochs_test = mne.Epochs(
            raw=self.test_data,
            events=self.events_test,
            event_id=self.event_id,
            tmin=0.0,
            tmax=self.tmax,
            baseline=None,
        )

        self.pipe = make_pipeline(
            FunctionTransformer(dp.eeg_power_band, validate=False),
            RandomForestClassifier(
                n_estimators=self.n_estimators_,
                min_samples_leaf=self.min_samples_leaf_,
                max_features=self.max_features_,
                random_state=self.random_state_,
            ),
        )

        self.y_train = self.epochs_train.events[:, 2]
        self.pipe.fit(self.epochs_train, self.y_train)

        self.y_pred = self.pipe.predict(self.epochs_test)

        self.y_test = self.epochs_test.events[:, 2]
        self.acc = accuracy_score(self.y_test, self.y_pred)
        return (
            self.train_data.info,
            self.test_data.info,
            self.n_estimators_,
            self.min_samples_leaf_,
            self.max_features_,
            self.random_state_,
            self.acc,
        )

        # plt.show()
