# main.py
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
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ----------------------------
# Configuration - Edit if needed
# ----------------------------
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

MODEL_PATH = "model.h5"  # change to your model filename if different
GOOGLE_DRIVE_FILE_ID = "1jEaQSeUgmelqsI4XRcna3AxR02bXplrb"  # if you host model in Drive
IMAGE_SIZE = (224, 224)  # model input size (width, height)
LAST_CONV_LAYER_NAME_GUESS = None  # if known, set e.g. "conv2d_7" else function will try to find one

# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

safe_mkdir(FEEDBACK_DIR)

# ----------------------------
# Load or download model
# ----------------------------
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("Model not found locally — attempting to download from Google Drive...")
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Model downloaded.")
        except Exception as e:
            st.error(f"Could not download model automatically: {e}")
            return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model file ({MODEL_PATH}): {e}")
        return None

model = load_ai_model()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image_pil(image: Image.Image, target_size=IMAGE_SIZE):
    image_resized = image.resize(target_size)
    arr = img_to_array(image_resized).astype("float32") / 255.0
    # ensure shape (1, H, W, C)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr

# ----------------------------
# Attempt to find last conv layer name if not provided
# ----------------------------
def get_last_conv_layer_name(m):
    # Try to find a conv layer by type/name scanning backwards
    for layer in reversed(m.layers):
        if "conv" in layer.name or "Conv" in layer.__class__.__name__:
            return layer.name
    # fallback to None
    return None

# ----------------------------
# Grad-CAM (best effort)
# ----------------------------
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Returns a heatmap (H, W) uint8 [0..255] aligned to the model input size.
    Best-effort: works for many keras models. last_conv_layer_name can be provided.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        raise ValueError("Could not determine last convolutional layer name. Please set LAST_CONV_LAYER_NAME_GUESS.")

    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # support both single-output and multi-output
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            # assume binary interest (take max logit)
            loss = tf.reduce_max(predictions, axis=-1)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None - gradcam failed. Check model and last conv layer.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)

    max_val = np.max(heatmap.numpy()) if hasattr(heatmap, "numpy") else np.max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    # resize heatmap to image size
    heatmap_np = heatmap.numpy() if hasattr(heatmap, "numpy") else np.array(heatmap)
    heatmap_resized = cv2.resize(heatmap_np, (img_array.shape[2], img_array.shape[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    return heatmap_uint8

# ----------------------------
# Simple retraining from saved feedback images
# ----------------------------
def retrain_model(feedback_dir=FEEDBACK_DIR, batch_size=4, epochs=3):
    if model is None:
        st.warning("No model loaded — cannot retrain.")
        return
    images, labels = [], []
    for fname in os.listdir(feedback_dir):
        if fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg"):
            # expected filename format: img_<rand>_label_<0_or_1>.png
            try:
                label = int(fname.split("_label_")[1].split(".")[0])
            except Exception:
                continue
            img = Image.open(os.path.join(feedback_dir, fname)).convert("RGB")
            images.append(preprocess_image_pil(img)[0])
            labels.append(label)
    if len(images) == 0:
        st.info("No feedback images found to retrain on.")
        return
    X = np.array(images)
    y = np.array(labels)
    try:
        model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(MODEL_PATH)
        st.success("Model updated with feedback.")
    except Exception as e:
        st.error(f"Retraining failed: {e}")

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Artificial Medical Intel", layout="centered")
st.title("🩺 Artificial Medical Intel — Breast Scan + Biomarkers")

st.markdown(
    "Upload a breast scan image and (optionally) enter blood biomarkers. "
    "This tool provides an AI-assisted assessment — **not** a diagnosis. Always consult a clinician."
)

# ----- Patient Info -----
st.subheader("🧑‍⚕️ Patient Information")
patient_name   = st.text_input("Patient Name", "")
patient_id     = st.text_input("Patient ID / Record Number", "")
patient_age    = st.number_input("Age", min_value=0, max_value=120, value=0)
patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
doctor_name    = st.text_input("Referring Doctor / Radiologist", "")
hospital_name  = st.text_input("Hospital / Facility", "")

st.write("---")

# ----- Image Upload -----
uploaded_file = st.file_uploader("📤 Upload scan image (mammogram / CT / ultrasound)", type=["jpg","jpeg","png"])

# ----- Biomarkers -----
st.subheader("🧪 Blood Test / Tumor Markers (optional)")
ca15_3 = st.number_input("CA 15-3 (U/mL)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
ca27_29 = st.number_input("CA 27-29 (U/mL)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
er = st.number_input("Estrogen Receptor (ER %)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
pr = st.number_input("Progesterone Receptor (PR %)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
her2 = st.selectbox("HER2 Expression", ["Unknown", "0", "1+", "2+", "3+"])
wbc = st.number_input("WBC (×10^9/L)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
rbc = st.number_input("RBC (×10^12/L)", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
platelets = st.number_input("Platelets (×10^9/L)", min_value=0.0, max_value=2000.0, value=0.0, step=1.0)

st.write("---")

# Placeholder variables that will be populated after analysis
result_text = "No analysis yet"
severity = "N/A"
confidence = 0.0
prediction = 0.0
biomarker_score = 0.0
final_score = 0.0
metrics_text = ""
scan_path = None
heatmap_path = None
overlay_path = None

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Unable to open uploaded file: {e}")
        image = None

    if image is not None:
        st.image(image, caption="Uploaded Scan Preview", use_column_width=True)
        img_np = np.array(image)
        if np.mean(img_np) < 5:
            st.warning("Image appears mostly black — please upload a valid scan.")
        else:
            # Preprocess for model
            img_input = preprocess_image_pil(image)

            if model is None:
                st.error("AI model not loaded. Check model path or Drive ID.")
            else:
                # Predict
                try:
                    raw_pred = model.predict(img_input)
                    # handle shapes: (1,1), (1,), (1,N)
                    if np.array(raw_pred).ndim == 2 and raw_pred.shape[-1] == 1:
                        prediction = float(raw_pred[0][0])
                    elif np.array(raw_pred).ndim == 2 and raw_pred.shape[-1] > 1:
                        # multiclass - take highest class probability
                        prediction = float(np.max(raw_pred))
                    elif np.array(raw_pred).ndim == 1:
                        prediction = float(raw_pred[0])
                    else:
                        prediction = float(np.max(raw_pred))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    prediction = 0.0

                # Biomarker scoring (simple heuristic weighting)
                biomarker_score = 0.0
                # thresholds and contributions - tweakable
                if ca15_3 > 25: biomarker_score += 0.15
                if ca27_29 > 38: biomarker_score += 0.10
                if er > 70: biomarker_score += 0.05
                if pr > 60: biomarker_score += 0.05
                her2_mapping = {"Unknown": 0.0, "0": 0.0, "1+": 0.05, "2+": 0.15, "3+": 0.25}
                biomarker_score += her2_mapping.get(her2, 0.0)
                if wbc != 0 and (wbc < 3 or wbc > 11): biomarker_score += 0.05
                # final combined score
                final_score = min(prediction + biomarker_score, 1.0)
                confidence = final_score * 100.0

                # Interpret final_score into text severity
                if final_score > 0.85:
                    result_text = "High likelihood of tumor"
                    severity = "High"
                elif final_score > 0.6:
                    result_text = "Possible tumor detected"
                    severity = "Medium"
                else:
                    result_text = "No tumor detected"
                    severity = "Low"

                # Show results
                st.write(f"**Prediction (image model):** {(prediction*100):.2f}%")
                st.write(f"**Biomarker influence:** {(biomarker_score*100):.2f}%")
                st.write(f"**Final combined confidence:** {confidence:.2f}%")
                st.write(f"**Result:** {result_text} — **Severity:** {severity}")

                # Try Grad-CAM and overlay
                heatmap = None
                overlay_img = None
                try:
                    last_conv = LAST_CONV_LAYER_NAME_GUESS
                    heatmap = generate_gradcam_heatmap(model, img_input, last_conv)
                    # colorize heatmap and overlay on resized original
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    original_resized = cv2.cvtColor(np.array(image.resize(IMAGE_SIZE)), cv2.COLOR_RGB2BGR)
                    overlay_img = cv2.addWeighted(original_resized, 0.6, heatmap_color, 0.4, 0)
                    # Save previews for PDF
                    scan_path = "scan_uploaded_preview.png"
                    overlay_path = "heatmap_overlay_preview.png"
                    Image.fromarray(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)).save(scan_path)
                    cv2.imwrite(overlay_path, overlay_img)
                    # Show overlay
                    st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption="Heatmap Overlay", use_column_width=True)
                except Exception as e:
                    st.warning(f"Heatmap generation failed (this can happen for some models): {e}")

                # Tumor metrics from heatmap (if available)
                metrics_text = ""
                try:
                    if heatmap is not None:
                        thresh = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY)[1]
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        draw = ImageDraw.Draw(image)
                        total_area = 0
                        for i, cnt in enumerate(contours):
                            x, y, w, h = cv2.boundingRect(cnt)
                            area = w * h
                            total_area += area
                            center_x = x + w // 2
                            center_y = y + h // 2
                            radius = max(w, h) // 2
                            # Draw circle around tumor on original image (note: coordinates are relative to resized input)
                            # We drew overlay on resized image earlier; draw on original preview for display
                            # Calculate scale factor from IMAGE_SIZE to original image size for drawing
                            orig_w, orig_h = image.size
                            scale_x = orig_w / IMAGE_SIZE[0]
                            scale_y = orig_h / IMAGE_SIZE[1]
                            cx = int(center_x * scale_x)
                            cy = int(center_y * scale_y)
                            r = int(radius * max(scale_x, scale_y))
                            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="red", width=3)
                            crop = np.array(image.crop((max(0, cx-r), max(0, cy-r), min(orig_w, cx+r), min(orig_h, cy+r))))
                            avg_color = tuple(np.mean(crop.reshape(-1, 3), axis=0).astype(int)) if crop.size else (0,0,0)
                            metrics_text += f"- Tumor {i+1}: center=({cx},{cy}), radius={r}px, area={area}px, avg_color={avg_color}\n"
                        coverage_ratio = total_area / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
                        metrics_text += f"\n- Tumor coverage ratio (approx): {coverage_ratio:.6f} of resized scan area\n"
                        # Save circled preview for PDF
                        circled_path = "scan_circled_preview.png"
                        image.save(circled_path)
                    else:
                        circled_path = None
                except Exception as e:
                    st.warning(f"Tumor metrics calculation failed: {e}")
                    circled_path = None

                # Show circled image if created
                if circled_path and os.path.exists(circled_path):
                    st.image(Image.open(circled_path), caption="Detected Tumor Regions (circled)", use_column_width=True)
                    st.write("**Tumor Metrics:**")
                    st.text(metrics_text)
                else:
                    if metrics_text:
                        st.write("**Tumor Metrics:**")
                        st.text(metrics_text)

                # Save session state relevant items for PDF / feedback
                st.session_state['last_scan_path'] = scan_path if scan_path else None
                st.session_state['last_overlay_path'] = overlay_path if overlay_path else None
                st.session_state['last_circled_path'] = circled_path if circled_path else None
                st.session_state['last_prediction'] = prediction
                st.session_state['last_biomarker_score'] = biomarker_score
                st.session_state['last_final_score'] = final_score
                st.session_state['last_result_text'] = result_text
                st.session_state['last_severity'] = severity
                st.session_state['last_confidence'] = confidence
                st.session_state['last_metrics_text'] = metrics_text
                st.session_state['uploaded_file_name'] = uploaded_file.name

                # Feedback Buttons
                st.write("---")
                st.write("**Was the prediction correct?**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, correct"):
                        label_val = 1 if final_score > 0.6 else 0
                        fname = f"{FEEDBACK_DIR}/img_{np.random.randint(1_000_000)}_label_{label_val}.png"
                        # save the original uploaded (full size) for retraining
                        Image.open(uploaded_file).convert("RGB").save(fname)
                        st.success("Feedback saved — thank you. Retraining model...")
                        retrain_model()
                with col2:
                    if st.button("❌ No, wrong"):
                        correct_label = st.radio("Select correct label", options=["Healthy", "Tumor"])
                        if st.button("Save correct label"):
                            label_val = 0 if correct_label == "Healthy" else 1
                            fname = f"{FEEDBACK_DIR}/img_{np.random.randint(1_000_000)}_label_{label_val}.png"
                            Image.open(uploaded_file).convert("RGB").save(fname)
                            st.success("Correct label saved. Retraining model...")
                            retrain_model()

                # PDF Export
                st.write("---")
                st.subheader("📄 Export Report (PDF)")
                if st.button("Generate PDF Report"):
                    # prepare data for pdf
                    pdf_filename = f"BreastScan_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(os.getcwd(), pdf_filename)
                    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = []

                    story.append(Paragraph("<b>ARTIFICIAL MEDICAL INTELLIGENCE — BREAST SCAN REPORT</b>", styles["Title"]))
                    story.append(Spacer(1, 12))
                    story.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
                    story.append(Paragraph(f"<b>File:</b> {st.session_state.get('uploaded_file_name','-')}", styles["Normal"]))
                    story.append(Spacer(1, 10))

                    # patient
                    story.append(Paragraph("<b>Patient Details</b>", styles["Heading2"]))
                    story.append(Paragraph(f"Name: {patient_name or '-'}", styles["Normal"]))
                    story.append(Paragraph(f"Patient ID: {patient_id or '-'}", styles["Normal"]))
                    story.append(Paragraph(f"Age: {patient_age}", styles["Normal"]))
                    story.append(Paragraph(f"Gender: {patient_gender}", styles["Normal"]))
                    story.append(Paragraph(f"Referring Doctor: {doctor_name or '-'}", styles["Normal"]))
                    story.append(Paragraph(f"Hospital/Facility: {hospital_name or '-'}", styles["Normal"]))
                    story.append(Spacer(1, 12))

                    # images
                    story.append(Paragraph("<b>Uploaded Scan</b>", styles["Heading2"]))
                    if st.session_state.get('last_scan_path') and os.path.exists(st.session_state['last_scan_path']):
                        story.append(RLImage(st.session_state['last_scan_path'], width=300, height=300))
                    else:
                        story.append(Paragraph("No scan preview image available.", styles["Normal"]))
                    story.append(Spacer(1, 10))

                    story.append(Paragraph("<b>AI Heatmap Overlay</b>", styles["Heading2"]))
                    if st.session_state.get('last_overlay_path') and os.path.exists(st.session_state['last_overlay_path']):
                        story.append(RLImage(st.session_state['last_overlay_path'], width=300, height=300))
                    else:
                        story.append(Paragraph("No heatmap image available.", styles["Normal"]))
                    story.append(Spacer(1, 12))

                    # AI results
                    story.append(Paragraph("<b>AI Prediction Results</b>", styles["Heading2"]))
                    story.append(Paragraph(f"Diagnosis: {st.session_state.get('last_result_text','-')}", styles["Normal"]))
                    story.append(Paragraph(f"Severity: {st.session_state.get('last_severity','-')}", styles["Normal"]))
                    story.append(Paragraph(f"Final Confidence: {st.session_state.get('last_confidence',0.0):.2f}%", styles["Normal"]))
                    story.append(Paragraph(f"Image Model Score: {st.session_state.get('last_prediction',0.0)*100:.2f}%", styles["Normal"]))
                    story.append(Paragraph(f"Biomarker Influence: {st.session_state.get('last_biomarker_score',0.0)*100:.2f}%", styles["Normal"]))
                    story.append(Spacer(1, 12))

                    # biomarkers table
                    story.append(Paragraph("<b>Blood Biomarkers</b>", styles["Heading2"]))
                    biomarker_table = [
                        ["Test", "Value"],
                        ["CA 15-3", f"{ca15_3} U/mL"],
                        ["CA 27-29", f"{ca27_29} U/mL"],
                        ["Estrogen Receptor (ER)", f"{er}%"],
                        ["Progesterone Receptor (PR)", f"{pr}%"],
                        ["HER2 Expression", f"{her2}"],
                        ["WBC", f"{wbc} ×10^9/L"],
                        ["RBC", f"{rbc} ×10^12/L"],
                        ["Platelets", f"{platelets} ×10^9/L"]
                    ]
                    tbl = Table(biomarker_table, colWidths=[200, 200])
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                        ("ALIGN", (1, 0), (-1, -1), "CENTER")
                    ]))
                    story.append(tbl)
                    story.append(Spacer(1, 12))

                    # tumor metrics
                    story.append(Paragraph("<b>Tumor Metrics (AI-detected regions)</b>", styles["Heading2"]))
                    mtext = st.session_state.get('last_metrics_text','-')
                    if mtext and mtext.strip():
                        for line in mtext.split("\n"):
                            if line.strip():
                                story.append(Paragraph(line, styles["Normal"]))
                    else:
                        story.append(Paragraph("No tumor metrics available.", styles["Normal"]))
                    story.append(Spacer(1, 30))

                    # doctor signature
                    story.append(Paragraph("<b>Doctor / Facility Verification</b>", styles["Heading2"]))
                    story.append(Spacer(1, 40))
                    story.append(Paragraph("______________________________", styles["Normal"]))
                    story.append(Paragraph("Signature / Stamp", styles["Normal"]))

                    # build and offer download
                    try:
                        doc.build(story)
                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Download PDF Report", f, file_name=pdf_filename, mime="application/pdf")
                        st.success("PDF report generated.")
                    except Exception as e:
                        st.error(f"Failed creating PDF: {e}")


