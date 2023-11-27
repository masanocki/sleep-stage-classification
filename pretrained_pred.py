import mne
import matplotlib
import matplotlib.pyplot as plt
import data_prep as dp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib


class ModelTrainer:
    def __init__(
        self,
        test_raw_data_path,
        test_annotations_path,
    ):
        self.test_raw_data_path_ = test_raw_data_path
        self.test_annotations_path_ = test_annotations_path
        matplotlib.use("Agg")

    def predict(self):
        self.event_id = {
            "Sleep stage W": 1,
            "Sleep stage N1": 2,
            "Sleep stage N2": 3,
            "Sleep stage N3": 4,
            "Sleep stage R": 5,
        }

        self.test_data = dp.read_file(
            self.test_raw_data_path_, data_type="raw", drop=True
        )

        self.annot_test = mne.read_annotations(self.test_annotations_path_)
        self.test_data.set_annotations(self.annot_test)

        self.events_test, _ = mne.events_from_annotations(
            self.test_data, event_id=self.event_id, chunk_duration=30.0
        )
        self.tmax = 30.0 - 1.0 / self.test_data.info["sfreq"]
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

        self.model = joblib.load("sleep_stage_model.sav")

        self.y_pred = self.model.predict(self.epochs_test)

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
