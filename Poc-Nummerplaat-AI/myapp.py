import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import time
import os
from tensorflow.keras.models import load_model  # ✅ FIXED IMPORT

# --- STYLING ---
st.set_page_config(page_title="Numberplate Scanner", page_icon="🚗", layout="wide")


# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "keras_model.h5")
        labels_path = os.path.join(BASE_DIR, "labels.txt")

        model = load_model(model_path, compile=False)

        with open(labels_path, "r") as f:
            class_names = f.readlines()

        return model, class_names

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


model, class_names = load_keras_model()

# --- STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

if "scan_trigger" not in st.session_state:
    st.session_state.scan_trigger = 0

if "last_processed_hash" not in st.session_state:
    st.session_state.last_processed_hash = None


def reset_scan():
    st.session_state.scan_trigger += 1
    st.session_state.last_processed_hash = None


def add_to_history(image, country, confidence):
    st.session_state.history.insert(0, {
        "image": image,
        "country": country,
        "confidence": confidence,
        "time": time.strftime("%H:%M:%S")
    })
    st.session_state.history = st.session_state.history[:5]


def process_image(img):
    if model is None or class_names is None:
        return None, 0.0

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized

    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name_raw = class_names[index].strip()
    class_name = class_name_raw[2:] if len(class_name_raw) > 2 else class_name_raw
    confidence = float(prediction[0][index])

    return class_name, confidence


# --- UI ---
st.title("🚗 Numberplate Country Scanner")

mode = st.sidebar.radio("Navigation", ["📷 Webcam Scanner", "📁 File Upload"])

if mode == "📁 File Upload":
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img)

        with st.spinner("Analyzing..."):
            country, conf = process_image(img)

        if country:
            st.success(f"Country: {country}")
            st.info(f"Confidence: {conf * 100:.2f}%")