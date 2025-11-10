import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
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
# Streamlit UI
# ----------------------------
st.title("Artificial Medical Intel - Self-Learning Breast Cancer Detection")
st.write("Upload CT / Mammography / Sonar images for tumor detection.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array_exp)[0][0]

    # Display prediction
    if prediction > 0.7:
        st.warning(f"⚠ Tumor likely detected (confidence {prediction*100:.2f}%)")
    elif prediction > 0.5:
        st.info(f"⚠ Possible tumor detected (confidence {prediction*100:.2f}%)")
    else:
        st.success(f"✅ No tumor detected (confidence {(1-prediction)*100:.2f}%)")

    # Ask for user feedback
    st.write("Was the prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, correct"):
            # Save image and predicted label
            filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1e6)}_label_{int(round(prediction))}.png"
            image.save(filename)
            st.success("Feedback saved for future learning!")

    with col2:
        if st.button("❌ No, wrong"):
            # Ask user to enter correct label
            correct_label = st.radio("Select correct label", options=["Healthy", "Tumor"])
            if st.button("Save correct label"):
                label_val = 0 if correct_label=="Healthy" else 1
                filename = f"{FEEDBACK_DIR}/img_{np.random.randint(1e6)}_label_{label_val}.png"
                image.save(filename)
                st.success("Correct label saved for future retraining!")

# ----------------------------
# Optional retraining button (manual trigger)
# ----------------------------
if st.button("Retrain model with feedback"):
    import glob
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    st.info("Starting retraining with feedback data...")

    # Load feedback images
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

        # Simple retraining
        model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=3, batch_size=4)

        # Save updated model
        model.save("model.h5")
        st.success("Model retrained and saved!")


