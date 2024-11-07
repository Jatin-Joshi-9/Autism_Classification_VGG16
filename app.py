import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the saved model
model = load_model("VGG16_Autism_best_model.keras")

# Set the title of the app
st.title("Autism Spectrum Disorder Image Classifier")

# Prompt the user to upload an image file
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")  # Add a blank line

    # Preprocess the image to match model input shape
    img = image.resize((224, 224))  # Resize to 224x224 for VGG16
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Rescale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run the model prediction
    prediction = model.predict(img_array)

    # Interpret the result
    if prediction[0][0] < 0.5:
        st.write("The model predicts: **Autistic**")
    else:
        st.write("The model predicts: **Non-Autistic**")

    # Optionally, show the prediction confidence
