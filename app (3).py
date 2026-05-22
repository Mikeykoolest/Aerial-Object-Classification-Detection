
# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st

import numpy as np

from PIL import Image

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Optional YOLO
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Aerial Object Classification",

    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🛩️ Aerial Object Classification & Detection")

st.write("Upload an aerial image to classify Bird or Drone.")

# =========================================================
# LOAD CLASSIFICATION MODEL
# =========================================================

classifier_model = load_model("best_bird_drone_model.keras")

# =========================================================
# OPTIONAL YOLO MODEL
# =========================================================

# Uncomment if you trained YOLOv8
# yolo_model = YOLO("best.pt")

# =========================================================
# CLASS LABELS
# =========================================================

class_names = {

    0: "Bird 🐦",

    1: "Drone 🚁"
}

# =========================================================
# IMAGE PREPROCESSING FUNCTION
# =========================================================

def preprocess_image(image):

    image = image.resize((224, 224))

    image = np.array(image)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(

    "Upload an Image",

    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True
    )

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    prediction = classifier_model.predict(processed_image)

    confidence = float(prediction[0][0])

    # =====================================================
    # CLASSIFICATION RESULT
    # =====================================================

    if confidence > 0.5:

        predicted_class = 1

        final_confidence = confidence

    else:

        predicted_class = 0

        final_confidence = 1 - confidence

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.subheader("Prediction")

    st.success(f"{class_names[predicted_class]}")

    st.info(f"Confidence Score: {final_confidence:.2%}")

    # =====================================================
    # OPTIONAL YOLO DETECTION
    # =====================================================

    # Uncomment if YOLO model available

    """
    results = yolo_model(image)

    st.subheader("YOLOv8 Detection")

    detected_image = results[0].plot()

    st.image(
        detected_image,
        caption="Detection Result",
        use_container_width=True
    )
    """
