import os
import cv2
import numpy as np
from utils.preprocessing import extract_features

# Create a dummy blank image
dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

try:
    features = extract_features(dummy_img)
    print(f"Extraction successful!")
    print(f"Number of features extracted: {len(features)}")
    assert len(features) == 31, f"Expected 31 features, got {len(features)}"
    print("Test Passed: Preprocessing logic is intact.")
except Exception as e:
    print(f"Extraction Failed: {e}")

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore

try:
    model = Sequential([
        Input(shape=(31,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Test Passed: Keras ANN Model successfully built and compiled.")
except Exception as e:
    print(f"Model Building Failed: {e}")
