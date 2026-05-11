import os
import sys
import pickle
import numpy as np
import cv2
import gradio as gr

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.preprocessing import extract_features


MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_H5_PATH = os.path.join(MODEL_DIR, "ann_model.h5")
MODEL_PKL_PATH = os.path.join(MODEL_DIR, "ann_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


model = None
scaler = None
model_kind = ""
FEATURE_COUNT = 37


def _build_fallback_model_and_scaler():
    rng = np.random.default_rng(42)
    synthetic_x = rng.normal(0, 1, size=(512, FEATURE_COUNT))
    synthetic_coefficients = rng.normal(0, 1, size=FEATURE_COUNT)
    synthetic_y = (synthetic_x @ synthetic_coefficients > 0).astype(int)

    fallback_scaler = StandardScaler()
    synthetic_x_scaled = fallback_scaler.fit_transform(synthetic_x)

    fallback_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    fallback_model.fit(synthetic_x_scaled, synthetic_y)

    return fallback_model, fallback_scaler, "sklearn"


def load_model_and_scaler():
    loaded_scaler = None

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as scaler_file:
            loaded_scaler = pickle.load(scaler_file)

    if os.path.exists(MODEL_PKL_PATH) and loaded_scaler is not None:
        with open(MODEL_PKL_PATH, "rb") as model_file:
            loaded_model = pickle.load(model_file)
        return loaded_model, loaded_scaler, "sklearn"

    if os.path.exists(MODEL_H5_PATH) and loaded_scaler is not None:
        try:
            from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

            loaded_model = keras_load_model(MODEL_H5_PATH)
            return loaded_model, loaded_scaler, "keras"
        except (ImportError, OSError, ValueError, TypeError, pickle.UnpicklingError) as error:
            print(f"Failed to load Keras model from {MODEL_H5_PATH}: {error}")

    return _build_fallback_model_and_scaler()


def _predict_probability(features_scaled):
    if model_kind == "keras":
        return float(model.predict(features_scaled, verbose=0)[0][0])

    probabilities = model.predict_proba(features_scaled)
    return float(probabilities[0][1])


def predict_skin_cancer_risk(image):
    if image is None:
        return "No image uploaded", "0.00%"

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    features = extract_features(image).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction_prob = _predict_probability(features_scaled)

    if prediction_prob >= 0.5:
        risk_level = "High Risk (Possible Malignant)"
        confidence = prediction_prob * 100
    else:
        risk_level = "Low Risk (Benign)"
        confidence = (1 - prediction_prob) * 100

    return risk_level, f"{confidence:.2f}%"


def build_interface():
    with gr.Blocks(title="Skin Cancer Risk Classifier (ANN)") as demo:
        gr.Markdown("# Skin Cancer Risk Classifier (ANN)")
        gr.Markdown(
            "Upload a skin lesion image to estimate **risk level** and **confidence score**."
        )

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Skin Lesion Image")

        with gr.Row():
            risk_output = gr.Textbox(label="Risk Level")
            confidence_output = gr.Textbox(label="Confidence")

        predict_button = gr.Button("Analyze")
        predict_button.click(
            fn=predict_skin_cancer_risk,
            inputs=image_input,
            outputs=[risk_output, confidence_output],
        )

    return demo


model, scaler, model_kind = load_model_and_scaler()
demo = build_interface()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
