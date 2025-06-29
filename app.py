# app.py

import os
import gdown
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# --- Constants ---
MODEL_PATH = 'retinopathy_best_model.h5'
GOOGLE_DRIVE_FILE_ID = '1L7J4ZnU4mYfIWVLy9d-lBa-zvpOnwaSk'
IMG_SIZE = 224
LABELS = ["Not Diabetic", "Diabetic"]

# --- Download Model If Not Exists ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model already exists. Skipping download.")

# Download model if needed
download_model()

# --- Load the Model ---
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Preprocessing Function ---
def preprocess_image(img_pil):
    img_pil = img_pil.convert('RGB')
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_pil)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

# --- Prediction Function ---
def predict(image):
    if model is None:
        return {"Error": 1.0, "Model not loaded": 1.0}
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    return {
        "Diabetic": float(prediction),
        "Not Diabetic": float(1 - prediction)
    }

# --- Gradio Interface ---
title = "Diabetic Retinopathy Detection"
description = """
Upload a retinal scan PNG image to check for signs of Diabetic Retinopathy.
The model will classify the image as either 'Diabetic' or 'Not Diabetic' and provide a confidence score.
This tool is based on a fine-tuned ResNet50 model. **Disclaimer: This is a demo and not for medical diagnosis.**
"""
article = "<p style='text-align: center;'>Built with Keras, TensorFlow, and Gradio.</p>"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retinal Scan Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Result"),
    title=title,
    description=description,
    article=article
)

if __name__ == "__main__":
    if model:
        iface.launch(share=True)
