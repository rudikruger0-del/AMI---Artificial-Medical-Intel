import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import cv2

# ----------------------------
# Config
# ----------------------------
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)
MODEL_PATH = "model.h5"
GOOGLE_DRIVE_FILE_ID = "1jEaQSeUgmelqsI4XRcna3AxR02bXplrb"

# ----------------------------
# Load or download model
# ----------------------------
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
        st.success("Download complete!")
    model = load_model(MODEL_PATH)
    return model

model = load_ai_model()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(image, target_size=(224,224)):
    image = image.resize(target_size)
    img_array = img_to_array(image)/255.0
    return np.expand_dims(img_array, axis=0)

# ----------------------------
# Grad-CAM for Sequential model
# ----------------------------
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="conv2d_1"):
    _ = model(img_array, training=False)
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    return heatmap

# ----------------------------
# Self-learning / Retraining
# ----------------------------
def retrain_model(feedback_dir=FEEDBACK_DIR, batch_size=4, epochs=3):
    images, labels = [], []
    for file in os.listdir(feedback_dir):
        if file.endswith(".png"):
            label = int(file.split("_label_")[1].split(".")[0])
            image = Image.open(os.path.join(feedback_dir, file)).convert("RGB")
            images.append(preprocess_image(image)[0])
            labels.append(label)
    if len(images) == 0:
        return
    X = np.array(images)
    y = np.array(labels)
    model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(MODEL_PATH)
    st.success("Model updated with feedback!")

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Artificial Medical Intel - Breast Cancer Scan Analysis")
st.write("Upload CT / Mammography / Sonar scans for tumor detection and analysis.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Ignore mostly-black images
    if np.mean(img_np) < 5:
        st.warning("Image appears mostly black — please upload a proper scan.")
    else:
        st.image(image, caption="Uploaded Scan", use_column_width=True)

        img_array_exp = preprocess_image(image)

        # Prediction
        prediction = model.predict(img_array_exp)[0][0]
        confidence = prediction*100
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

        # Grad-CAM heatmap overlay
        heatmap = None
        try:
            heatmap = generate_gradcam_heatmap(model, img_array_exp)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            original_img = cv2.cvtColor(np.array(image.resize((224,224))), cv2.COLOR_RGB2BGR)
            overlay_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
            st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption="Heatmap Overlay", use_column_width=True)
        except Exception as e:
            st.warning(f"Heatmap could not be generated: {e}")

        # Tumor metrics with circles
        if heatmap is not None:
            try:
                thresh = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY)[1]
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                metrics_text = ""
                draw = ImageDraw.Draw(image)
                total_area = 0

                for i, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w*h
                    total_area += area
                    center_x = x + w // 2
                    center_y = y + h // 2
                    radius = max(w,h)//2

                    # Draw circle around tumor
                    draw.ellipse([(center_x - radius, center_y - radius),
                                  (center_x + radius, center_y + radius)], outline="red", width=3)

                    crop = np.array(image.crop((x, y, x+w, y+h)))
                    avg_color = tuple(np.mean(crop.reshape(-1,3), axis=0).astype(int))
                    metrics_text += f"- Tumor {i+1}: center=({center_x},{center_y}), radius={radius}px, area={area}px, avg_color={avg_color}\n"

                coverage_ratio = total_area / (img_np.shape[0] * img_np.shape[1])
                metrics_text += f"\n- Tumor coverage ratio: {coverage_ratio:.4f} of scan area\n"

                st.image(image, caption="Detected Tumor Regions (circled)", use_column_width=True)
                st.write("**Tumor Metrics:**")
                st.text(metrics_text)

            except Exception as e:
                st.warning(f"Tumor metrics could not be calculated: {e}")

        # Professional medical notes
        st.write("**Medical Analysis / Recommendations:**")
        if prediction > 0.6:
            st.write("""
- Immediate consultation with oncologist/radiologist advised.
- Additional imaging (MRI, CT, ultrasound) recommended.
- Biopsy may be indicated depending on clinical context.
- Review patient history and other scans.
- Follow-up schedule to be defined based on tumor severity.
""")
        else:
            st.write("""
- No evidence of tumor detected in this scan.
- Routine screening and follow-ups per medical guidelines.
""")

        # Feedback mechanism
        st.write("**Was the prediction correct?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, correct"):
                label_val = 1 if prediction > 0.6 else 0
                filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1_000_000)}_label_{label_val}.png"
                image.save(filename)
                st.success("Feedback saved! Updating model...")
                retrain_model()

        with col2:
            if st.button("❌ No, wrong"):
                correct_label = st.radio("Select correct label", options=["Healthy","Tumor"])
                if st.button("Save correct label"):
                    label_val = 0 if correct_label=="Healthy" else 1
                    filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1_000_000)}_label_{label_val}.png"
                    image.save(filename)
                    st.success("Correct label saved! Updating model...")
                    retrain_model()

