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
  <a href="https://huggingface.co/spaces/Broehit/skin-cancer-ann">
    <img src="https://img.shields.io/badge/🚀 Live%20Demo-Hugging%20Face-yellow?style=for-the-badge">
  </a>
</p>

<h1 align="center">Skin Cancer Risk Classifier (ANN)</h1>

<div align="center">
  <strong>Developed with ❤️ by Rohit Manal</strong>
</div>
<br/>

## ✅ Deployment Status

The application now runs on **Gradio** and is ready for Hugging Face Spaces deployment without requiring missing Flask templates or mandatory pre-saved model files.

## 🌐 Hugging Face Spaces Deployment

### Option 1: Gradio SDK (recommended)
1. Create a new Hugging Face Space.
2. Select **Gradio** SDK.
3. Push this repository.
4. Hugging Face will launch `app/app.py` automatically.

### Option 2: Docker SDK
1. Create a new Hugging Face Space.
2. Select **Docker** SDK.
3. Push this repository.
4. Hugging Face builds from `Dockerfile` and exposes port `7860`.

## 🧠 Inference Pipeline

- Upload image through Gradio UI.
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
python app/app.py
```

Then open: `http://127.0.0.1:7860`
