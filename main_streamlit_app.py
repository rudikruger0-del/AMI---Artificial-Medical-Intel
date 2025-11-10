import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
def load_ai_model():
    model = load_model("model.h5")  # Ensure model.h5 is in the same folder
    return model

model = None

st.title("Artificial Medical Intel - Breast Cancer Cell Detection")
st.write("Upload CT/Mammography/Sonar scan images to detect potential tumors.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            if model is None:
                model = load_ai_model()

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = prediction[0][0]

        if result > 0.5:
            st.error("⚠ Tumor detected. Recommend medical review.")
        else:
            st.success("✅ No tumor detected in the scanned region.")
