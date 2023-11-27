from pretrained_pred import ModelTrainer


clf = ModelTrainer(
    test_raw_data_path="./datasets/recordings/SN002.edf",
    test_annotations_path="./datasets/recordings/SN002_sleepscoring.edf",
)
clf.predict()
