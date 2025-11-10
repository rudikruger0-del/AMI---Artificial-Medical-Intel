import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import os
import gdown
import cv2
import glob
import tensorflow as tf

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
# Grad-CAM heatmap
# ----------------------------
def generate_heatmap(model, img_array, last_conv_layer_name="conv2d_1"):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
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
# Streamlit UI
# ----------------------------
st.title("Artificial Medical Intel - Self-Learning Breast Cancer Detection")
st.write("Upload CT / Mammography / Sonar images to detect potential tumors.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array_exp)[0][0]
    confidence = prediction*100

    # Display prediction and treatment suggestions
    if prediction > 0.7:
        st.warning(f"⚠ Tumor likely detected (confidence {confidence:.2f}%)")
        st.write("**Recommended next steps / possible treatment plans:**")
        st.write("- Schedule an appointment with an oncologist or radiologist")
        st.write("- Consider further imaging (MRI, CT, or ultrasound)")
        st.write("- Discuss possible biopsy for confirmation")
        st.write("- Begin early treatment planning if confirmed by doctor")
    elif prediction > 0.5:
        st.info(f"⚠ Possible tumor detected (confidence {confidence:.2f}%)")
        st.write("**Recommended next steps:**")
        st.write("- Consult your doctor for a second opinion")
        st.write("- Monitor with follow-up imaging")
        st.write("- Further tests may be needed to confirm")
    else:
        st.success(f"✅ No tumor detected (confidence {100-confidence:.2f}%)")
        st.write("**Recommended next steps:**")
        st.write("- Routine screening as per medical guidelines")
        st.write("- Maintain healthy lifestyle and regular check-ups")

    # Generate heatmap
    try:
        heatmap = generate_heatmap(model, img_array_exp, last_conv_layer_name="conv2d_1")
        heatmap_img = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)
        st.image(heatmap_img, caption="Heatmap of tumor areas", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")

    # ----------------------------
    # Feedback section
    # ----------------------------
    st.write("Was the prediction correct?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, correct"):
            label_val = 1 if prediction > 0.5 else 0
            filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1e6)}_label_{label_val}.png"
            image.save(filename)
            st.success("Feedback saved for future learning!")

    with col2:
        if st.button("❌ No, wrong"):
            correct_label = st.radio("Select correct label", options=["Healthy", "Tumor"])
            if st.button("Save correct label"):
                label_val = 0 if correct_label=="Healthy" else 1
                filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1e6)}_label_{label_val}.png"
                image.save(filename)
                st.success("Correct label saved for future retraining!")

# ----------------------------
# Manual retraining button
# ----------------------------
if st.button("Retrain model with feedback"):
    st.info("Starting retraining with feedback data...")
    files = glob.glob(f"{FEEDBACK_DIR}/*.png")
    if len(files) == 0:
        st.warning("No feedback data found.")
    else:
        X = []
        y = []
        for f in files:
            img = Image.open(f).resize((224,224))
            X.append(np.array(img)/255.0)
            label = int(f.split("_label_")[1].split(".png")[0])
            y.append(label)
        X = np.array(X)
        y = np.array(y)

        # Retrain model
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=3, batch_size=4)
        model.save("model.h5")
        st.success("Model retrained and saved!")


