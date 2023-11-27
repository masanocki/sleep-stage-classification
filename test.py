from model_pred import ModelPred

clf = ModelPred(
    train_raw_data_path="./datasets/recordings/SN001.edf",
    train_annotations_path="./datasets/recordings/SN001_sleepscoring.edf",
    test_raw_data_path="./datasets/recordings/SN002.edf",
    test_annotations_path="./datasets/recordings/SN002_sleepscoring.edf",
    n_estimators=100,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
)

clf.predict()
