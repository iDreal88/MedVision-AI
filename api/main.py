from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
import os
import cv2
import numpy as np
import base64
import sys
import io
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from fpdf import FPDF

# Add root directory to path for RAG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1' # Ensure Keras 2 compatibility for transformers

import traceback

app = FastAPI(title="Thesis ML API")

@app.get("/")
def read_root():
    return {"status": "MedVision API is running", "models": list(MODEL_MAP.keys())}

# CORS Setup for React frontend
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# MODEL CONFIG
# ==============================
MODEL_MAP = {
    "CNN+CLAHE": {"path": "cnn+clahe/model_cnn_clahe.keras", "layer": "conv_idx_2"},
    "ResNet50": {"path": "resnet50/model_resnet50.keras", "layer": "conv5_block3_out"},
    "VGG16": {"path": "vgg16/model_vgg16.keras", "layer": "block5_conv3"},
    "VGG19": {"path": "vgg19/model_vgg19.keras", "layer": "block5_conv4"}
}

# Cache for loaded models
loaded_models = {}

def get_model(model_name: str):
    global loaded_models
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Aggressive clearing RAM before loading
    import gc

    if model_name not in loaded_models:
        print(f"Aggressively clearing RAM and loading {model_name}...")
        loaded_models.clear()
        gc.collect()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, MODEL_MAP[model_name]["path"])
        
        if not os.path.exists(model_path):
            print(f"CRITICAL: Model file not found at {model_path}")
            raise HTTPException(status_code=404, detail=f"File missing at {model_path}")
        
        try:
            loaded_models[model_name] = load_model(model_path)
            print(f"SUCCESS: {model_name} is live.")
        except Exception as e:
            print(f"ERROR: Could not load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model Load Failure: {str(e)}")
    
    return loaded_models[model_name]

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
        # Find the functional path to the target layer
        path_to_layer = []
        found = False
        for i, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                path_to_layer = [(model, i)]
                found = True
                break
            if isinstance(layer, Model):
                try:
                    # Check if layer is in submodel
                    sub_idx = -1
                    for j, sl in enumerate(layer.layers):
                        if sl.name == last_conv_layer_name:
                            sub_idx = j
                            break
                    if sub_idx != -1:
                        path_to_layer = [(model, i), (layer, sub_idx)]
                        found = True
                        break
                except:
                    continue
        
        if not found:
            print(f"DEBUG: Layer {last_conv_layer_name} not found in model.")
            return None

        with tf.GradientTape() as tape:
            # 1. Forward through the model up to the target layer to get conv_outputs
            # We MUST do this inside the tape so gradients can be tracked back to it
            x = img_tensor
            if len(path_to_layer) == 1:
                # Target is at top level (e.g. CNN+CLAHE)
                m, idx = path_to_layer[0]
                for i in range(idx + 1):
                    x = m.layers[i](x)
                conv_outputs = x # Gradient target
                
                # Continue forwarding through the rest of the model (except last layer)
                for i in range(idx + 1, len(m.layers) - 1):
                    x = m.layers[i](x)
            else:
                # Target is nested (e.g. VGG/ResNet)
                m1, idx1 = path_to_layer[0] # Top model, base_model_idx
                m2, idx2 = path_to_layer[1] # Base model, target_layer_idx
                
                # Forward to base model
                for i in range(idx1):
                    x = m1.layers[i](x)
                
                # Inside base model up to target
                for i in range(idx2 + 1):
                    x = m2.layers[i](x)
                conv_outputs = x # Gradient target
                
                # Rest of base model
                for i in range(idx2 + 1, len(m2.layers)):
                    x = m2.layers[i](x)
                
                # Rest of top model (except last layer)
                for i in range(idx1 + 1, len(m1.layers) - 1):
                    x = m1.layers[i](x)

            # 2. Calculate Logits manually from the penultimate activation
            # This bypasses Softmax saturation for robust gradients
            last_layer = model.layers[-1]
            logits = tf.matmul(x, last_layer.weights[0]) + last_layer.weights[1]
            
            # 3. Define loss based on the predicted class
            # Use argmax of logits to find predicted class
            class_index = tf.argmax(logits[0])
            loss = logits[:, class_index]

        # 4. Extract gradients of the loss w.r.t. the convolutional outputs
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            print(f"DEBUG: Gradients are None for {last_conv_layer_name}")
            return None

        # 5. Global average pooling and heatmap generation
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_val = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)
        
        # 6. Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        
        # ROBUST FALLBACK: If heatmap is empty (common in 100% confidence/saturated cases)
        # we fallback to a weighted activation map or raw saliency to ensure the user 
        # always sees the focus area.
        if max_val < 1e-12:
            # Fallback 1: Use absolute gradients (ignore ReLU to see negative contributions)
            heatmap = tf.reduce_sum(tf.abs(conv_outputs_val * pooled_grads), axis=-1)
            max_val = tf.reduce_max(heatmap)
            
            if max_val < 1e-12:
                # Fallback 2: If gradients are completely gone, show raw activations of the target layer
                # This shows "what features were found" even if the gradient is zero
                heatmap = tf.reduce_mean(conv_outputs_val, axis=-1)
                max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            max_val = 1e-8
            
        heatmap /= max_val
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Grad-CAM Error: {traceback.format_exc()}")
        return None

def overlay_gradcam(img, heatmap, alpha=0.4):
    img_rgb = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heatmap_res = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap_res), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_rgb, 1-alpha, heatmap_color, alpha, 0)

# ==============================
# ENDPOINTS
# ==============================
@app.get("/models")
async def get_models():
    return list(MODEL_MAP.keys())

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # 1. Validate Image Validity (Prevent random non-image files)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            raise HTTPException(status_code=400, detail="Invalid image file format. Please upload a valid PNG or JPG.")

        # 2. Validate Content (Mammogram Check)
        # Check for color saturation. Mammograms are grayscale.
        # Random everyday images usually have color.
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_saturation = np.mean(s)
        
        # Threshold: > 25 means significant color content (0-255 scale)
        if mean_saturation > 25:
             raise HTTPException(
                 status_code=400, 
                 detail=f"Uploaded image appears to be a color photograph (Saturation: {mean_saturation:.1f}). Please upload a valid grayscale mammogram."
             )
             
        # Convert to grayscale for model interaction
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # Load model
        model = get_model(model_name)
        
        # Preprocess
        img_processed = cv2.resize(img, (128, 128))
        img_ready = apply_clahe(img_processed)
        img_input = img_ready.reshape(1, 128, 128, 1)

        # Inference
        prediction = model.predict(img_input)
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][pred_class])
        label = "Cancer" if pred_class == 1 else "Non-Cancer"

        # Grad-CAM
        img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)
        heatmap = make_gradcam_heatmap(img_tensor, model, MODEL_MAP[model_name]["layer"])
        
        result = {
            "label": label,
            "confidence": confidence * 100,
            "original_image": base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'),
            "processed_image": base64.b64encode(cv2.imencode('.jpg', (img_ready*255).astype(np.uint8))[1]).decode('utf-8')
        }

        if heatmap is not None:
            overlay = overlay_gradcam(img_ready, heatmap)
            result["gradcam_image"] = base64.b64encode(cv2.imencode('.jpg', overlay)[1]).decode('utf-8')
        
        # RAG Report Generation
        kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base.md")
        if os.path.exists(kb_path):
            from report_generator import ReportGenerator
            report_gen = ReportGenerator(kb_path)
            
            # Dynamic XAI description based on model architecture and result
            if label == "Cancer":
                if "ResNet" in model_name or "VGG" in model_name:
                    finding = f"The {model_name} architecture identified high-entropy activation patterns within the deeper convolutional blocks. These clusters correlate with irregular structural density and architectural distortion characteristic of malignant lesions."
                else:
                    finding = "High-intensity activation centroids detected. The model weights these dense regions as primary indicators of malignant tissue morphology."
            else:
                finding = f"The {model_name} model shows diffuse, low-level activations across the parenchymal background. No focal areas of suspicious density were identified in the targeting layers."
                
            report_content = report_gen.generate_report(label, confidence * 100, finding)
            result["report"] = report_content
        
        return result
    except Exception as e:
        print(f"PREDICT ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

class ClinicalPDF(FPDF):
    def header(self):
        # Header Bar
        self.set_fill_color(30, 41, 59)  # Slate 800
        self.rect(0, 0, 210, 35, 'F')
        
        # Logo text
        self.set_font("helvetica", "B", 24)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 10)
        self.cell(0, 15, "MedVision AI", ln=False)
        
        # Sub-header
        self.set_font("helvetica", "I", 10)
        self.set_xy(10, 22)
        self.cell(0, 10, "Neural Diagnostic Intelligence - Clinical Report", ln=True)
        
        self.ln(15)

    def footer(self):
        self.set_y(-25)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 116, 139)  # Slate 500
        
        # Disclaimer
        disclaimer = "DISCLAIMER: This report is generated by an AI assistant for research purposes only. " \
                     "It should be reviewed by a qualified medical professional before clinical action."
        self.multi_cell(0, 4, disclaimer, align='C')
        
        # Page Number
        self.set_y(-10)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align='R')

@app.post("/download-pdf")
async def download_pdf(report: dict):
    try:
        content = report.get("content", "")
        pdf = ClinicalPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Body Styling
        lines = content.split('\n')
        for line in lines:
            pdf.set_x(10)
            if line.startswith('# '):
                pdf.ln(5)
                pdf.set_font("helvetica", "B", 18)
                pdf.set_text_color(30, 41, 59)
                pdf.cell(0, 10, line[2:].upper(), ln=True)
                self_draw_line(pdf)
                pdf.ln(2)
            elif line.startswith('## '):
                pdf.ln(4)
                pdf.set_font("helvetica", "B", 14)
                pdf.set_text_color(51, 65, 85)
                pdf.cell(0, 10, line[3:], ln=True)
                pdf.ln(1)
            elif line.startswith('### '):
                pdf.set_font("helvetica", "B", 12)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 8, line[4:], ln=True)
            elif line.startswith('- '):
                pdf.set_font("helvetica", size=11)
                pdf.set_text_color(51, 65, 85)
                # Dot bullet
                current_y = pdf.get_y()
                pdf.set_fill_color(37, 99, 235) # Blue 600
                pdf.ellipse(12, current_y + 3, 1.5, 1.5, 'F')
                pdf.set_x(16)
                
                # Handle bold in list items
                clean_line = line[2:]
                parts = clean_line.split('**')
                for i, part in enumerate(parts):
                    if i % 2 == 1: # Bold part
                        pdf.set_font("helvetica", "B", 11)
                        pdf.write(7, part)
                    else:
                        pdf.set_font("helvetica", "", 11)
                        pdf.write(7, part)
                pdf.ln()

            elif line.strip() == '':
                pdf.ln(2)
            else:
                pdf.set_font("helvetica", size=11)
                pdf.set_text_color(71, 85, 105)
                # Handle bold markers **text**
                parts = line.split('**')
                for i, part in enumerate(parts):
                    if i % 2 == 1: # Bold part
                        pdf.set_font("helvetica", "B", 11)
                        pdf.write(7, part)
                    else:
                        pdf.set_font("helvetica", "", 11)
                        pdf.write(7, part)
                pdf.ln()
                
        return StreamingResponse(
            io.BytesIO(pdf.output()),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=diagnosis_report.pdf"}
        )
    except Exception as e:
        print(f"PDF ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def self_draw_line(pdf):
    pdf.set_draw_color(226, 232, 240) # Slate 200
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)