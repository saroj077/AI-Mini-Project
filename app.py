# app.py

import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# --- Constants and Model Loading ---
MODEL_PATH = 'retinopathy_best_model.h5'
IMG_SIZE = 224
LABELS = ["Not Diabetic", "Diabetic"]

# Load the trained model once when the script starts
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'retinopathy_best_model.h5' is in the same directory.")
    # Exit or handle the error appropriately
    model = None

# --- Preprocessing and Prediction Functions ---

def preprocess_image(img_pil):
    """
    Preprocesses a PIL image to be ready for the ResNet50 model.
    1. Converts to RGB (if it has an alpha channel).
    2. Resizes to the required input size.
    3. Converts to a NumPy array.
    4. Adds a batch dimension.
    5. Applies ResNet50-specific preprocessing.
    """
    img_pil = img_pil.convert('RGB')  # Ensure 3 channels
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_pil)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 224, 224, 3)
    
    # Use the same preprocessing as during training
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

def predict(image):
    """
    The main prediction function that will be wrapped by Gradio.
    Takes a PIL image, preprocesses it, makes a prediction, and formats the output.
    """
    if model is None:
        return {"Error": 1.0, "Model not loaded": 1.0}

    # Preprocess the input image
    processed_image = preprocess_image(image)
    
    # Get the model's prediction
    prediction = model.predict(processed_image)[0][0]
    
    # The output of sigmoid is the probability of the positive class (Diabetic)
    # Gradio's Label component expects a dictionary of labels and their confidences.
    confidence_diabetic = float(prediction)
    confidence_not_diabetic = 1 - confidence_diabetic
    
    # Format for Gradio Label output
    output_data = {
        "Diabetic": confidence_diabetic,
        "Not Diabetic": confidence_not_diabetic
    }
    
    return output_data

# --- Gradio Interface ---
# Define the UI components and layout
title = "Diabetic Retinopathy Detection"
description = """
Upload a retinal scan PNG image to check for signs of Diabetic Retinopathy.
The model will classify the image as either 'Diabetic' or 'Not Diabetic' and provide a confidence score.
This tool is based on a fine-tuned ResNet50 model. **Disclaimer: This is a demo and not for medical diagnosis.**
"""
article = "<p style='text-align: center;'>Built with Keras, TensorFlow, and Gradio.</p>"

# The gr.Image component will handle the image upload. type="pil" provides a PIL image to our function.
# The gr.Label component is perfect for displaying classification results.
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retinal Scan Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Result"),
    title=title,
    description=description,
    article=article,
    examples=[
        # You can add example images here. They should be in a subfolder.
        # For example: [['examples/sample1.png']]
    ]
)

# --- Launch the App ---
if __name__ == "__main__":
    if model:
        # launch() creates a local web server. share=True generates a public link for 72 hours.
        iface.launch(share=True)