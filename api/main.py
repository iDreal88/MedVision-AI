from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import base64
import sys
import io
import gc
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(title="MedVision-AI Production API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

loaded_models = {}

def get_model(model_name):
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model_name not in loaded_models:
        loaded_models.clear()
        gc.collect()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, MODEL_MAP[model_name]["path"])
        
        try:
            loaded_models[model_name] = load_model(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model Load Failure: {str(e)}")
    
    return loaded_models[model_name]

# ==============================
# UTILITIES
# ==============================
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = clahe.apply(img)
    return img_c / 255.0

def generate_gradcam(model, img_array, layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

# ==============================
# ENDPOINTS
# ==============================
@app.get("/")
def read_root():
    return {"status": "PRODUCTION_LIVE", "agent_engine": "GEMINI_3_FLASH"}

@app.get("/models")
def list_models():
    return list(MODEL_MAP.keys())

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            raise HTTPException(status_code=400, detail="Invalid Image")

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        model = get_model(model_name)
        
        # Preprocessing
        img_resized = cv2.resize(img_gray, (128, 128))
        img_clahe = apply_clahe(img_resized)
        img_input = img_clahe.reshape(1, 128, 128, 1)

        # Inference
        prediction = model.predict(img_input)
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][pred_class])
        label = "Cancer" if pred_class == 1 else "Non-Cancer"

        # Grad-CAM
        heatmap = generate_gradcam(model, img_input, MODEL_MAP[model_name]["layer"], pred_class)
        
        # Prepare Images for JSON
        _, buffer_proc = cv2.imencode('.jpg', (img_clahe * 255).astype(np.uint8))
        proc_base64 = base64.b64encode(buffer_proc).decode('utf-8')

        gradcam_base64 = None
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (128, 128))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            img_bg = cv2.cvtColor((img_clahe * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cam_img = cv2.addWeighted(img_bg, 0.6, heatmap_color, 0.4, 0)
            _, buffer_cam = cv2.imencode('.jpg', cam_img)
            gradcam_base64 = base64.b64encode(buffer_cam).decode('utf-8')

        return {
            "label": label,
            "confidence": confidence * 100,
            "processed_image": proc_base64,
            "gradcam_image": gradcam_base64,
            "report": f"## Patient/Case Information\n- AI Diagnosis: {label}\n- Statistical Confidence: {confidence*100:.2f}%\n- Methodology: {model_name}"
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent-research")
async def agent_research(
    label: str = Form(...),
    confidence: float = Form(...),
    model_name: str = Form(...),
    technical_summary: str = Form(...)
):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API Key missing on server")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview")

        kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base.md")
        kb_context = ""
        if os.path.exists(kb_path):
            with open(kb_path, "r") as f:
                kb_context = f.read()

        prompt = f"""
        Role: Senior Resident Oncologist & AI Researcher.
        Findings: {label} ({confidence:.2f}%), Model: {model_name}
        Context: {kb_context}
        Technical Summary: {technical_summary}
        Task: Provide a clinical oncology synthesis correlating these AI findings with standard BI-RADS criteria.
        """
        
        response = model.generate_content(prompt)
        return {"agent_report": response.text}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
