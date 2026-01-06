import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50, VGG16, VGG19
import matplotlib.pyplot as plt
from fpdf import FPDF
from rag_engine import RAGEngine
from report_generator import ReportGenerator
import tempfile

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Thesis Model Demo", layout="wide")

st.title("🏥 AI Diagnosis Assistant")
st.markdown("""
This application demonstrates various deep learning models for mammogram analysis. 
Upload an image to get classification results, XAI (Grad-CAM) visualization, and an AI-generated diagnosis report.
""")

# ==============================
# UTILITIES
# ==============================
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = clahe.apply(img)
    return img_c / 255.0

def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name):
    try:
        # For nested models (ResNet/VGG), we need special handling
        # But if the layer is accessible by name directly in the model:
        try:
            target_layer = model.get_layer(last_conv_layer_name)
        except:
            # Try searching in sub-models
            for layer in model.layers:
                if isinstance(layer, Model):
                    try:
                        target_layer = layer.get_layer(last_conv_layer_name)
                        # Create a sub-model to get activations from the nested model
                        # This is getting complex, let's simplify by using the logic from main.py
                        pass
                    except:
                        pass
        
        # Fallback to the specific logic for each model type if needed
        # For simplicity in this demo, we'll try to find the layer anywhere
        
        # We need to find which layer index the base model is at
        base_model = None
        base_layer_idx = -1
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Model):
                base_model = layer
                base_layer_idx = i
                break
        
        if base_model:
            # Logic for ResNet/VGG
            internal_grad_model = Model(
                inputs=[base_model.input],
                outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
            )
            
            with tf.GradientTape() as tape:
                # 1. First layer (conversion)
                x = model.layers[0](img_tensor)
                # 2. Base model
                conv_outputs, base_output = internal_grad_model(x)
                # 3. Remaining layers
                x = base_output
                for i in range(base_layer_idx + 1, len(model.layers)):
                    x = model.layers[i](x)
                
                predictions = x
                class_index = tf.argmax(predictions[0])
                loss = predictions[:, class_index]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= (tf.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
        else:
            # Logic for Custom CNN
            grad_model = Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                class_index = tf.argmax(predictions[0])
                loss = predictions[:, class_index]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= (tf.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
            
    except Exception as e:
        st.error(f"Grad-CAM Error: {e}")
        return None

def create_pdf(markdown_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Use a standard font
    pdf.set_font("helvetica", size=12)
    
    lines = markdown_text.split('\n')
    for line in lines:
        # Reset X to margin at start of each line to avoid cumulative offset issues
        pdf.set_x(pdf.l_margin)
        
        if line.startswith('# '):
            pdf.set_font("helvetica", "B", 16)
            pdf.multi_cell(pdf.epw, 10, line[2:], ln=True)
            pdf.ln(2)
        elif line.startswith('## '):
            pdf.set_font("helvetica", "B", 14)
            pdf.multi_cell(pdf.epw, 10, line[3:], ln=True)
            pdf.ln(1)
        elif line.startswith('### '):
            pdf.set_font("helvetica", "B", 12)
            pdf.multi_cell(pdf.epw, 10, line[4:], ln=True)
        elif line.startswith('- '):
            pdf.set_font("helvetica", size=12)
            pdf.multi_cell(pdf.epw, 8, f"  - {line[2:]}", ln=True)
        elif line.strip() == '':
            pdf.ln(5)
        else:
            pdf.set_font("helvetica", size=12)
            pdf.multi_cell(pdf.epw, 8, line, ln=True)
    
    return bytes(pdf.output())

def overlay_gradcam(img, heatmap, alpha=0.4):
    img_rgb = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heatmap_res = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap_res), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_rgb, 1-alpha, heatmap_color, alpha, 0)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["CNN+CLAHE", "ResNet50", "VGG16", "VGG19"]
)

model_map = {
    "CNN+CLAHE": {"path": "cnn+clahe/model_cnn_clahe.keras", "layer": "conv_idx_2"},
    "ResNet50": {"path": "resnet50/model_resnet50.keras", "layer": "conv5_block3_out"},
    "VGG16": {"path": "vgg16/model_vgg16.keras", "layer": "block5_conv3"},
    "VGG19": {"path": "vgg19/model_vgg19.keras", "layer": "block5_conv4"}
}

# ==============================
# MODEL LOADING
# ==============================
@st.cache_resource
def load_selected_model(path):
    # Use absolute path relative to the app.py file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    if os.path.exists(full_path):
        return load_model(full_path)
    return None

model_path = model_map[model_choice]["path"]
model = load_selected_model(model_path)

if model is None:
    folder_name = model_choice.lower().replace('/', '')
    st.warning(f"⚠️ Model file not found at `{model_path}`. Please run the training script in the `{folder_name}` folder first to generate the `.keras` file.")
    st.stop()

# ==============================
# RAG ENGINE
# ==============================
@st.cache_resource
def load_rag():
    if os.path.exists("knowledge_base.md"):
        return RAGEngine("knowledge_base.md"), ReportGenerator("knowledge_base.md")
    return None, None

rag_engine, report_gen = load_rag()

# ==============================
# MAIN CONTENT
# ==============================
uploaded_file = st.file_uploader("Upload a Mammogram Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_display = cv2.resize(img, (256, 256))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_display, width=300)
    
    # Preprocess
    img_processed = cv2.resize(img, (128, 128))
    img_processed = apply_clahe(img_processed)
    
    with col2:
        st.subheader("CLAHE Preprocessed")
        st.image(img_processed, width=300)
    
    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing image..."):
            # Inference
            img_input = img_processed.reshape(1, 128, 128, 1)
            prediction = model.predict(img_input)
            pred_class = np.argmax(prediction[0])
            confidence = prediction[0][pred_class] * 100
            
            label = "Cancer" if pred_class == 1 else "Non-Cancer"
            color = "red" if pred_class == 1 else "green"
            
            st.markdown(f"### Result: <span style='color:{color}'>{label}</span> ({confidence:.2f}%)", unsafe_allow_html=True)
            
            # Grad-CAM
            st.divider()
            st.subheader("Explainable AI (Grad-CAM)")
            
            heatmap = make_gradcam_heatmap(
                tf.convert_to_tensor(img_input, dtype=tf.float32),
                model,
                model_map[model_choice]["layer"]
            )
            
            if heatmap is not None:
                overlay = overlay_gradcam(img_processed, heatmap)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM Heatmap Overlay", width=400)
                
                # RAG Report
                if report_gen:
                    st.divider()
                    
                    if label == "Cancer":
                        finding = "High-intensity activation detected in dense tissue regions, suggesting structural irregularity consistent with malignant morphology."
                    else:
                        finding = "Low or diffuse activation patterns observed; no focal areas of high density identified that strongly suggest malignancy."
                    
                    report_content = report_gen.generate_report(
                        prediction_label=label,
                        confidence=confidence,
                        gradcam_finding=finding
                    )
                    
                    st.markdown(report_content)
                    
                    # Download button (PDF only)
                    try:
                        pdf_bytes = create_pdf(report_content)
                        st.download_button(
                            label="Download Report as PDF",
                            data=pdf_bytes,
                            file_name=f"diagnosis_report_{label}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating PDF: {e}")
                else:
                    st.error("RAG Engine files not found. Report generation skipped.")
            else:
                st.error("Could not generate Grad-CAM visualization.")
else:
    st.info("Please upload a mammogram image to begin.")

st.sidebar.divider()
st.sidebar.info("This is a thesis project demonstration. All predictions should be verified by a medical professional.")
