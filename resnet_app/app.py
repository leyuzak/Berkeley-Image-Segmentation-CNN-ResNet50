import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (224, 224)
MODEL_PATH = "resnet50_model.keras"
THRESH = 0.10

model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="ResNet50 Segmentation", layout="centered")
st.title("🐟 Fish Segmentation - ResNet50")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    img = cv2.resize(image_np, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    mask = (pred.squeeze() > THRESH).astype(np.uint8) * 255

    mask_resized = cv2.resize(
        mask,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = image_np.copy()
    overlay[mask_resized == 255] = [255, 0, 0]

    st.subheader("Original Image")
    st.image(image_np, use_column_width=True)

    st.subheader("Predicted Mask")
    st.image(mask_resized, clamp=True, use_column_width=True)

    st.subheader("Overlay")
    st.image(overlay, use_column_width=True)
