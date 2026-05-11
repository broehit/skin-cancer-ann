import numpy as np

from utils.preprocessing import extract_features
from app.app import load_model_and_scaler, predict_skin_cancer_risk


def test_feature_extraction_shape():
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    features = extract_features(dummy_img)
    expected_feature_count = 6 + 6 + (8 * 3) + 1
    assert len(features) == expected_feature_count, f"Expected {expected_feature_count} features, got {len(features)}"


def test_fallback_model_pipeline_runs():
    loaded_model, loaded_scaler, loaded_model_kind = load_model_and_scaler()
    assert loaded_model is not None
    assert loaded_scaler is not None
    assert loaded_model_kind in {"keras", "sklearn"}


if __name__ == "__main__":
    test_feature_extraction_shape()
    test_fallback_model_pipeline_runs()

    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    risk, confidence = predict_skin_cancer_risk(dummy_img)

    assert risk in {"Low Risk (Benign)", "High Risk (Possible Malignant)"}
    assert confidence.endswith("%")
    print("All checks passed.")
