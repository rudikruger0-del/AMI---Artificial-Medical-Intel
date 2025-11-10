import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import gdown
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# ----------------------------
# Load AI model with caching
# ----------------------------
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

model = load_ai_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Artificial Medical Intel - Breast Cancer Detection")
st.write("Upload CT / Mammography / Sonar scan images to detect potential tumors.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ----------------------------
# Grad-CAM function
# ----------------------------
def generate_heatmap(model, img_array, last_conv_layer_name="conv2d_1"):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# ----------------------------
# Prediction and visualization
# ----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            # Preprocess image
            img_resized = image.resize((224, 224))
            img_array = img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            result = prediction[0][0]  # value between 0 and 1
            confidence = result * 100

            # Display message and recommended next steps
            if result > 0.7:
                st.error(f"⚠ Tumor likely detected. Confidence: {confidence:.2f}%")
                st.write("**Recommended next steps:**")
                st.write("- Schedule an appointment with an oncologist or radiologist")
                st.write("- Consider further imaging (MRI, CT, or ultrasound)")
                st.write("- Discuss possible biopsy for confirmation")
                st.write("- Begin early treatment planning if confirmed by doctor")
            elif result > 0.5:
                st.warning(f"⚠ Possible tumor detected. Confidence: {confidence:.2f}%")
                st.write("**Recommended next steps:**")
                st.write("- Consult your doctor for a second opinion")
                st.write("- Monitor with follow-up imaging")
                st.write("- Further tests may be needed to confirm")
            else:
                st.success(f"✅ No tumor detected. Confidence: {100 - confidence:.2f}%")
                st.write("**Recommended next steps:**")
                st.write("- Routine screening as per medical guidelines")
                st.write("- Maintain healthy lifestyle and regular check-ups")

            # Generate heatmap
            try:
                heatmap = generate_heatmap(model, img_array, last_conv_layer_name="conv2d_1")
                heatmap_img = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)
                st.image(heatmap_img, caption="Heatmap of tumor areas", use_column_width=True)
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")

