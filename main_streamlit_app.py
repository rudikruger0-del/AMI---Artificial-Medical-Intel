import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import cv2

# ----------------------------
# Load AI model
# ----------------------------
@st.cache_resource
def load_ai_model():
    model_path = "model.h5"
    file_id = "1jEaQSeUgmelqsI4XRcna3AxR02bXplrb"
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
# Grad-CAM function
# ----------------------------
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="conv2d_1"):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    return heatmap

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Artificial Medical Intel - Breast Cancer Scan Analysis")
st.write("Upload CT / Mammography / Sonar scans to detect potential tumors.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert to numpy array for black mask checking
    img_np = np.array(image)
    if np.mean(img_np) < 5:  # mostly black image
        st.warning("Image appears mostly black or empty — please upload a proper scan.")
    else:
        st.image(image, caption="Uploaded Scan", use_column_width=True)

        # Preprocess image
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)/255.0
        img_array_exp = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array_exp)[0][0]
        confidence = prediction * 100

        # Grad-CAM heatmap
        heatmap = None
        try:
            heatmap = generate_gradcam_heatmap(model, img_array_exp, last_conv_layer_name="conv2d_1")
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            original_img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
            overlay_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
            st.image(overlay_pil, caption="Heatmap Overlay (model-focused)", use_column_width=True)
        except Exception as e:
            st.warning(f"Heatmap could not be generated: {e}")

        # Safe thresholds
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

        # Tumor metrics
        if heatmap is not None:
            try:
                thresh = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY)[1]
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                metrics_text = ""
                draw = ImageDraw.Draw(image)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    crop = np.array(image.crop((x, y, x + w, y + h)))
                    avg_color = tuple(np.mean(crop.reshape(-1, 3), axis=0).astype(int))
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                    metrics_text += f"- Tumor region: x={x}, y={y}, width={w}, height={h}, area={area} px, avg_color={avg_color}\n"

                st.image(image, caption="Detected Tumor Regions with Bounding Boxes", use_column_width=True)
                if metrics_text:
                    st.write("**Tumor Metrics:**")
                    st.text(metrics_text)
                else:
                    st.write("No distinct tumor regions detected for metric estimation.")
            except Exception as e:
                st.warning(f"Tumor metrics could not be calculated: {e}")
        else:
            st.info("Tumor metrics skipped because heatmap could not be generated.")

        # Professional medical notes
        st.write("**Medical Analysis / Recommendations:**")
        if prediction > 0.6:
            st.write("""
- Immediate consultation with oncologist or radiologist is advised.
- Consider further imaging (MRI, CT, ultrasound) for confirmation.
- Biopsy may be recommended based on clinical judgment.
- Review patient history and other scans.
- Follow-up schedule should be defined based on tumor severity.
""")
        else:
            st.write("""
- No evidence of tumor detected in this scan.
- Routine screening and follow-ups per medical guidelines recommended.
- Encourage healthy lifestyle and regular check-ups.
""")

        # Feedback mechanism
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

