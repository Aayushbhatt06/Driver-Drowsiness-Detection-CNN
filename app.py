import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import os
import subprocess

# Load your CNN model
model = load_model("driver_drowsiness_cnn.h5")
labels = ['awake', 'sleepy']

# Streamlit page setup
st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Upload an eye/face image **or** use your camera for real-time detection.")

# Sidebar for options
mode = st.sidebar.radio("Select Mode", ["🖼️ Upload Image", "📷 Live Camera"])

# ------------- UPLOAD MODE -------------
if mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        # Predict
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction)
        result = labels[pred_class]

        # Display
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if result == "awake":
            st.success("✅ Driver is Awake (Eyes Open)")
        else:
            st.error("⚠️ Driver is Drowsy (Eyes Closed)")

        st.write("Prediction Probabilities:", prediction)

# ------------- LIVE CAMERA MODE -------------
elif mode == "📷 Live Camera":
    st.write("Click below to start the **real-time detection window** (opens separately).")

    if st.button("▶ Start Live Camera"):
        # Save temporary live script
        live_script = """
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('driver_drowsiness_cnn.h5')
labels = ['awake', 'sleepy']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (64, 64))
            eye_img = eye_img / 255.0
            eye_img = np.expand_dims(eye_img, axis=(0, -1))

            prediction = model.predict(eye_img, verbose=0)
            pred_class = np.argmax(prediction)
            result = labels[pred_class]

            color = (0, 255, 0) if result == "awake" else (0, 0, 255)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            cv2.putText(frame, result, (x+ex, y+ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
        # Write temporary script
        with open("live_camera.py", "w") as f:
            f.write(live_script)

        # Run it in a subprocess (separate window)
        subprocess.Popen(["python", "live_camera.py"], shell=True)
        st.success("✅ Live camera started! (Press 'q' in window to stop)")
