import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
import data_prep as dp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class ModelPred:
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
        matplotlib.use("Agg")

    def add_additional_training_data(
        self, additional_raw_data_path, additional_annotations_path
    ):
        # Wczytaj dodatkowe dane treningowe
        additional_train_data = dp.read_file(
            additional_raw_data_path, data_type="raw", drop=True
        )
        additional_annot_train = mne.read_annotations(additional_annotations_path)
        additional_train_data.set_annotations(additional_annot_train)

        # Utwórz nowe zdarzenia
        additional_events_train, _ = mne.events_from_annotations(
            additional_train_data, event_id=self.event_id, chunk_duration=30.0
        )

        # Utwórz nowe epoki treningowe
        additional_epochs_train = mne.Epochs(
            raw=additional_train_data,
            events=additional_events_train,
            event_id=self.event_id,
            tmin=0.0,
            tmax=self.tmax,
            baseline=None,
        )

        # Dopisz nowe dane treningowe do modelu
        self.epochs_train = mne.concatenate_epochs(
            [self.epochs_train, additional_epochs_train]
        )

        # Naucz model na nowych danych treningowych
        self.pipe.fit(self.epochs_train, self.epochs_train.events[:, 2])

    def predict(self):
        self.train_data = dp.read_file(
            self.train_raw_data_path_, data_type="raw", drop=True
        )

        self.annot_train = mne.read_annotations(self.train_annotations_path_)

        self.train_data.set_annotations(self.annot_train)

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
        # self.train_fig = self.train_data.plot(
        #     events=self.events_train,
        #     scalings=dict(eeg=3e-5, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1),
        #     event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y"},
        # )

        # self.train_events_fig = mne.viz.plot_events(
        #     self.events_train,
        #     event_id=self.event_id,
        #     sfreq=self.train_data.info["sfreq"],
        #     first_samp=self.events_train[0, 0],
        # )

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
        # self.test_fig = self.test_data.plot(
        #     events=self.events_test,
        #     scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1),
        #     event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y"},
        # )
        # self.test_events_fig = mne.viz.plot_events(
        #     self.events_test,
        #     event_id=self.event_id,
        #     sfreq=self.test_data.info["sfreq"],
        #     first_samp=self.events_test[0, 0],
        # )
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
        self.add_additional_training_data(
            additional_raw_data_path="./datasets/recordings/SN003.edf",
            additional_annotations_path="./datasets/recordings/SN003_sleepscoring.edf",
        )

        self.y_pred = self.pipe.predict(self.epochs_test)

        self.y_test = self.epochs_test.events[:, 2]
        self.acc = accuracy_score(self.y_test, self.y_pred)
        print(self.acc)
        # self.report = classification_report(
        #     self.y_test,
        #     self.y_pred,
        #     target_names=self.event_id.keys(),
        #     output_dict=True,
        # )

        # self.confusion_matrix = confusion_matrix(
        #     self.y_test,
        #     self.y_pred,
        # )

        # self.conf_matrix_plot = ConfusionMatrixDisplay(
        #     confusion_matrix=self.confusion_matrix,
        #     display_labels=[
        #         "Sleep stage W",
        #         "Sleep stage N1",
        #         "Sleep stage N2",
        #         "Sleep stage N3",
        #         "Sleep stage R",
        #     ],
        # ).plot()

        # create array of events based on predicted labels
        # self.predicted_events = np.column_stack(
        #     (self.events_test[:, 0], np.zeros_like(self.y_pred), self.y_pred)
        # )

        # self.predicted_events_fig = mne.viz.plot_events(
        #     self.predicted_events,
        #     event_id=self.event_id,
        #     sfreq=self.epochs_test.info["sfreq"],
        # )

        # return (
        #     self.train_data.info,
        #     self.test_data.info,
        #     self.n_estimators_,
        #     self.min_samples_leaf_,
        #     self.max_features_,
        #     self.random_state_,
        #     self.acc,
        #     self.report,
        #     self.train_fig,
        #     self.test_fig,
        #     self.conf_matrix_plot.figure_,
        #     self.train_events_fig,
        #     self.test_events_fig,
        #     self.predicted_events_fig,
        # )
