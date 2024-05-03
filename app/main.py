import os
import json
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"E:\\project_24\\medNet\\plant-disease-prediction-cnn-deep-leanring-project-main\\app\\trained_model\\MedNet_Custom.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(180, 180)):
    # Load the image
    img = tf.keras.utils.load_img(
      image_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array

# Function to Predict the Class of an Image
# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    score = tf.nn.softmax(predictions[0])

    predicted_class_index = np.argmax(score)
    print(predicted_class_index)
    # Check if the predicted class index falls within the range of plant classes
    confidence_threshold = 0.5  # Adjust this threshold as needed
    
    # Check if the confidence score for the predicted class is above the threshold
    if score[predicted_class_index] >= confidence_threshold:
        predicted_class_name = class_indices[str(predicted_class_index)]
        return predicted_class_name
    else:
        return "Not a plant"


# Streamlit App
st.title('Medicinal Plant Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((180, 180))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
