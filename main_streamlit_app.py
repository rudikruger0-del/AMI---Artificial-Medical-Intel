import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image, ImageDraw
import numpy as np
import os
import gdown
import matplotlib.pyplot as plt
import cv2

# ----------------------------
# Load AI model
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
# Feedback storage
# ----------------------------
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Artificial Medical Intel - Breast Cancer Scan Analysis")
st.write("Upload CT / Mammography / Sonar scans for tumor detection.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    # ----------------------------
    # Preprocess image
    # ----------------------------
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # Model prediction
    # ----------------------------
    prediction = model.predict(img_array_exp)[0][0]
    confidence = prediction * 100

    # ----------------------------
    # Generate heatmap
    # ----------------------------
    try:
        from tensorflow.keras.models import Model
        last_conv_layer = model.layers[0]  # first Conv2D layer
        heatmap_model = Model(inputs=model.inputs, outputs=last_conv_layer.output)
        conv_output = heatmap_model.predict(img_array_exp)[0]
        heatmap = np.mean(conv_output, axis=-1)
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap / np.max(heatmap))
        heatmap_img = Image.fromarray(heatmap).convert("RGB")
        heatmap_img = Image.blend(image, heatmap_img, alpha=0.5)
        st.image(heatmap_img, caption="Heatmap Overlay", use_column_width=True)
    except:
        st.warning("Heatmap could not be generated.")

    # ----------------------------
    # Safe thresholds
    # ----------------------------
    if prediction > 0.85:
        result_text = "Tumor detected"
        severity = "High"
    elif prediction > 0.6:
        result_text = "Possible tumor detected"
        severity = "Medium"
    else:
        result_text = "No tumor detected"
        severity = "Low"

    st.write(f"**Prediction:** {result_text}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Severity rating:** {severity}")

    # ----------------------------
    # Tumor metrics estimation (bounding box)
    # ----------------------------
    try:
        gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        metrics_text = ""
        draw = ImageDraw.Draw(image)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w*h
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
            metrics_text += f"- Tumor region: x={x}, y={y}, width={w}, height={h}, area={area} pixels\n"

        st.image(image, caption="Detected Tumor Regions", use_column_width=True)
        if metrics_text:
            st.write("**Tumor Metrics:**")
            st.text(metrics_text)
        else:
            st.write("No clear tumor regions detected for metric estimation.")
    except:
        st.warning("Could not estimate tumor metrics.")

    # ----------------------------
    # Professional medical-style notes
    # ----------------------------
    st.write("**Medical Analysis / Recommendations:**")
    if prediction > 0.6:
        st.write("""
- Immediate consultation with oncologist or radiologist is advised.
- Consider further imaging (MRI, CT, ultrasound) for confirmation.
- Biopsy may be recommended based on clinical judgment.
- Patient history and other scans should be considered.
""")
    else:
        st.write("""
- No evidence of tumor detected in this scan.
- Routine screening and follow-ups per guidelines recommended.
- Encourage healthy lifestyle and regular check-ups.
""")

    # ----------------------------
    # Feedback mechanism for self-learning
    # ----------------------------
    st.write("Was the prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, correct"):
            label_val = 1 if prediction > 0.6 else 0
            filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1_000_000)}_label_{label_val}.png"
            image.save(filename)
            st.success("Feedback saved for future learning!")

    with col2:
        if st.button("❌ No, wrong"):
            correct_label = st.radio("Select correct label", options=["Healthy", "Tumor"])
            if st.button("Save correct label"):
                label_val = 0 if correct_label=="Healthy" else 1
                filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1_000_000)}_label_{label_val}.png"
                image.save(filename)
                st.success("Correct label saved for future retraining!")
