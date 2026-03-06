---
title: Skin Cancer Risk Classifier (ANN)
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

<h1 align="center">Skin Cancer Risk Classifier (ANN)</h1>

<div align="center">
  <strong>Developed with ❤️ by Rohit Manal</strong>
</div>
<br/>

<div align="center">
  <p>An end-to-end Machine Learning web application powered by an Artificial Neural Network, designed to classify skin lesions as <strong>Benign (Low Risk)</strong> or <strong>Malignant (High Risk)</strong> based on clinical imaging data.</p>
</div>

---

## 🌐 Live Demo & 24x7 Deployment

The application is fully configured for a **24x7 cloud deployment** using [Hugging Face Spaces](https://huggingface.co/spaces). Due to the memory footprint of Artificial Neural Networks, Hugging Face's Docker Spaces (16GB RAM) represent the most reliable free hosting.

### How to Host it 24x7 for Free:
1. **Fork or Push** this repository to your own GitHub account.
2. Sign up on [Hugging Face](https://huggingface.co/) and click **New Space**.
3. Name your space, select **Docker** as the Space SDK, and choose **Blank**.
4. Inside the Space settings, connect your GitHub repository or simply drag and drop the files.
5. Hugging Face will automatically detect the provided `Dockerfile` configuration and build your container.
6. Once the build finishes, your app will be securely hosted 24x7 with a permanent URL!

*(Note: The `render.yaml` and `Procfile` are still included in the repo if you wish to try Render or Heroku, but Hugging Face provides infinitely more reliable ML hosting).*

---

## 🧠 Project Overview
Detecting skin cancer early saves lives. This project demonstrates a complete, production-ready AI pipeline: 
1. **Feature Extraction**: Processing images from the clinical HAM10000 Skin Cancer Dataset to extract rich tabular features (Color means, Standard deviations, RGB histograms, and Canny edge densities).
2. **Model Training**: A highly-optimized dense Artificial Neural Network (ANN) built from scratch using Keras/TensorFlow.
3. **Inference & UI**: A modern, responsive, glassmorphic web interface built in Flask to serve the model's predictions in real-time.

## ⚙️ Model Architecture & Performance
The core engine is a Keras Sequential Deep Neural Network architected for advanced tabular feature analysis:
- **Input Features**: 31 extracted color & texture dimensions normalized via StandardScaler.
- **Hidden Layers**: 
  - Dense (128 units, ReLU)
  - Dense (64 units, ReLU)
  - Dense (32 units, ReLU)
  - Dropout Layer (0.3 rate) applied for robust regularization.
- **Output Layer**: Dense (1 unit, Sigmoid) acting as the binary classifier.
- **Performance Evaluation**: Successfully achieved **~82.3% Validation Accuracy** using the Adam Optimizer and Binary Crossentropy loss over 10,015 clinical images.

## 🛠️ Technology Stack
- **Artificial Intelligence**: Python, TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, OpenCV
- **Backend Infrastructure**: Flask, Werkzeug
- **Frontend UI/UX**: HTML5, Vanilla CSS3 (Glassmorphism aesthetics), Javascript, Phosphor Icons
