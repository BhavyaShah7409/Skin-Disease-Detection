import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Loading the model
MODEL_PATH = "model.keras"
model = load_model(MODEL_PATH)

# Defining class labels
CLASS_LABELS = {
    0: 'Eczema',
    1: 'Melanoma',
    2: 'Atopic',
    3: 'Basal',
    4: 'Melanocytic',
    5: 'Benign',
    6: 'Psoriasis',
    7: 'Seborrheic',
    8: 'Tinea',
    9: 'Warts'
}

# Input image dimensions
IMG_HEIGHT = 75
IMG_WIDTH = 100

# Preprocessing the image
def preprocess_image(image_path):
    # Loading the image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Converting to numpy array
    img_array = img_to_array(img)
    # Normalizing pixel values
    img_array = img_array / 255.0
    # Adding batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title("Skin Disease Detection App")
st.write("Upload an image of a skin condition, and the app will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # temporary path for image
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)
    
    # Preprocessing the image
    preprocessed_image = preprocess_image(temp_image_path)

    # Predicting using the model
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = CLASS_LABELS[predicted_class_index]

    # Displaying the prediction
    st.write(f"**Predicted Disease:** {predicted_label}")

    # Cleaning up temporary file
    os.remove(temp_image_path)

else:
    st.write("Please upload an image")
