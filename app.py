import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("driver_drowsiness_cnn.h5")

# Class labels (from your training setup)
labels = ['awake', 'sleepy']

# Streamlit App Title
st.title("🚗 Driver Drowsiness Detection")
st.write("Upload an image of the driver's eyes to check if they are open or closed.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape (1, 64, 64, 1)

    # Make prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    result = labels[pred_class]

    # Display result
    if result == 'awake':
        st.success("✅ Driver is Awake (Eyes Open)")
    else:
        st.error("⚠️ Driver is Drowsy (Eyes Closed)")

    # Show raw prediction probabilities
    st.write("Prediction probabilities:", prediction)