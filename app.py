import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Leaf Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Leaf Disease Classification")
st.write(
    "Upload a clear photo of a plant leaf. The model will predict one of the "
    "categories it learned during training."
)
st.warning(
    "Educational prototype only. A prediction is not a confirmed diagnosis or treatment advice. "
    "Results can be wrong, especially for photos unlike the training dataset."
)

MODEL_PATH = "plant_leaf_model.keras"
LABELS_PATH = "class_names.json"
IMG_SIZE = (160, 160)

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    st.error(
        "Model files are missing. Run `python train_model.py` first, then restart this app."
    )
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        resized = image.resize(IMG_SIZE)
        image_array = np.asarray(resized, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)

        if st.button("Classify leaf", type="primary"):
            probabilities = model.predict(image_batch, verbose=0)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]
            best_index = int(top_indices[0])
            best_label = class_names[best_index].replace("___", " — ").replace("_", " ")

            st.subheader("Model prediction")
            st.success(f"{best_label}")
            st.write(f"Model confidence score: {probabilities[best_index] * 100:.1f}%")
            st.caption(
                "This score is the model's output confidence, not the probability that a real plant "
                "has this disease."
            )

            st.subheader("Top predictions")
            for idx in top_indices:
                label = class_names[int(idx)].replace("___", " — ").replace("_", " ")
                st.write(f"**{label}** — {probabilities[idx] * 100:.1f}%")
                st.progress(float(probabilities[idx]))
    except Exception as exc:
        st.error(f"Could not read this image. Please try a JPG or PNG file. Details: {exc}")

with st.expander("About the model"):
    st.write(
        "This project uses transfer learning with MobileNetV2. The pretrained image feature "
        "extractor is frozen, and a new classification layer is trained on the leaf dataset."
    )
