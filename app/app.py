import os
import sys
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model # type: ignore

# Add parent directory to sys.path so we can import utils.preprocessing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.preprocessing import extract_features

app = Flask(__name__)

# Paths for models
MODEL_PATH = os.path.join(BASE_DIR, "model", "ann_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Global variables to hold the loaded model and scaler
model = None
scaler = None

def load_models():
    """Load model and scaler into memory."""
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = load_model(MODEL_PATH)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            print("Model and Scaler loaded successfully.")
        else:
            print("Warning: Model or Scaler not found. Train the model first.")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models at application startup
load_models()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded on server."}), 500
        
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading."}), 400
        
    try:
        # Read image to memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Uploaded file is not a valid image."}), 400
            
        # Extract tabular features using our utils module
        features = extract_features(img)
        
        # Reshape to 2D array for scaling and prediction
        features_2d = features.reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_2d)
        
        # Predict using the loaded ANN
        prediction_prob = model.predict(features_scaled)[0][0]
        
        # Interpret result
        if prediction_prob > 0.5:
            risk = "High Risk (Possible Malignant)"
            confidence = float(prediction_prob * 100)
        else:
            risk = "Low Risk (Benign)"
            confidence = float((1 - prediction_prob) * 100)
            
        return jsonify({
            "risk_level": risk,
            "confidence": round(confidence, 2)
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
