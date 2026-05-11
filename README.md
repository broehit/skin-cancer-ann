---
title: Skin Cancer Risk Classifier (ANN)
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app/app.py
pinned: false
---

<p align="center">
  <a href="https://huggingface.co/spaces/Broehit/skin-cancer-ann-v2">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face%20Spaces-blue?style=for-the-badge">
  </a>
</p>

<h1 align="center">Skin Cancer Risk Classifier (ANN)</h1>

<div align="center">
  <strong>Developed with ❤️ by Rohit Manal</strong>
</div>
<br/>

## 🚀 Quick Start

Click the button above to try the live classifier with our modern, tech-gradient UI!

## ✅ Deployment Status

The application now runs on **Flask** with a custom modern UI and is deployed on **Hugging Face Spaces** with a beautiful glassmorphism design and gradient styling.

## 🌐 Live Deployment

**Live URL:** https://huggingface.co/spaces/Broehit/skin-cancer-ann-v2

The app features:
- 🎨 Modern glassmorphism UI with gradient backgrounds
- 📸 Drag-and-drop image upload
- ⚡ Real-time AI predictions
- 📊 Confidence score with progress visualization
- 🎯 Risk level classification with color-coded results

## 🧠 Inference Pipeline

- Upload image through the custom web UI.
- Features are extracted via `utils/preprocessing.py` (`extract_features`).
- The app tries to load:
  - `model/ann_model.h5` + `model/scaler.pkl`, or
  - `model/ann_model.pkl` + `model/scaler.pkl`
- If model files are missing, a fallback ANN-like model and scaler are created automatically so the demo still works.
- Output includes:
  - **Risk Level** (`Low Risk (Benign)` / `High Risk (Possible Malignant)`)
  - **Confidence** (`xx.xx%`)

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open: `http://127.0.0.1:5000`

## 📁 Project Structure

```
skin-cancer-ann/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── templates/
│   └── index.html        # Custom UI template
├── static/
│   └── style.css         # Modern styling
└── utils/
    └── preprocessing.py  # Feature extraction
```

## 🔧 Tech Stack

- **Backend:** Flask, scikit-learn
- **Frontend:** HTML5, CSS3 (Glassmorphism, Gradients)
- **Model:** Artificial Neural Network (sklearn MLPClassifier)
- **Deployment:** Hugging Face Spaces
