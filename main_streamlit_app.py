import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

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
st.title("Artificial Medical Intel - Reliable Breast Cancer Detection")
st.write("Upload CT / Mammography / Sonar scans for tumor detection.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    # Preprocess image: Resize and normalize
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array_exp)[0][0]
    confidence = prediction * 100

    # Safe thresholds
    if prediction > 0.85:
        st.error(f"⚠ Tumor detected (confidence {confidence:.2f}%)")
        st.write("**Recommended next steps / treatment suggestions:**")
        st.write("- Schedule appointment with oncologist or radiologist")
        st.write("- Consider further imaging (MRI, CT, ultrasound)")
        st.write("- Discuss possible biopsy for confirmation")
    elif prediction > 0.6:
        st.warning(f"⚠ Possible tumor detected (confidence {confidence:.2f}%)")
        st.write("**Recommended next steps:**")
        st.write("- Consult your doctor for second opinion")
        st.write("- Monitor with follow-up imaging")
    else:
        st.success(f"✅ No tumor detected (confidence {100-confidence:.2f}%)")
        st.write("**Recommended next steps:**")
        st.write("- Routine screening per medical guidelines")
        st.write("- Maintain healthy lifestyle and regular check-ups")

    # ----------------------------
    # Feedback mechanism
    # ----------------------------
    st.write("Was the prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, correct"):
            label_val = 1 if prediction > 0.6 else 0
            filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1e6)}_l_
