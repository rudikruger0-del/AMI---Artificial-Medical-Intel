import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# Function to load AI model with caching
@st.cache_resource
def load_ai_model():
    model_path = "model.h5"
    file_id = "1jEaQSeUgmelqsI4XRcna3AxR02bXplrb"  # Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        st.info("Downloading AI model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        st.success("Download complete!")

    model = load_model(model_path)
    return model

# Load model
model = load_ai_model()

# Streamlit UI
st.title("Artificial Medical Intel - Breast Cancer Cell Detection")
st.write("Upload CT/Mammography/Sonar scan images to detect potential tumors.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = prediction[0][0]

        if result > 0.5:
            st.error("⚠ Tumor detected. Recommend medical review.")
        else:
            st.success("✅ No tumor detected in the scanned region.")
