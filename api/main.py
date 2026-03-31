from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
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

# ==============================
# ENDPOINTS
# ==============================
@app.get("/")
def read_root():
    return {"status": "PRODUCTION_LIVE", "agent_engine": "GEMINI_3_FLASH"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            raise HTTPException(status_code=400, detail="Invalid Image")

        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        model = get_model(model_name)
        
        img_processed = cv2.resize(img, (128, 128))
        img_ready = apply_clahe(img_processed)
        img_input = img_ready.reshape(1, 128, 128, 1)

        prediction = model.predict(img_input)
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][pred_class])
        label = "Cancer" if pred_class == 1 else "Non-Cancer"

        return {
            "label": label,
            "confidence": confidence * 100,
            "report": f"Automated diagnosis completed using {model_name}. Final classification: {label}."
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
    """
    Exposes the MedVision-Agent's research capabilities to the frontend.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API Key missing on server")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview")

        # Load Knowledge Base text
        kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base.md")
        kb_context = ""
        if os.path.exists(kb_path):
            with open(kb_path, "r") as f:
                kb_context = f.read()

        prompt = f"""
        Role: Senior Resident Oncologist & AI Researcher.
        
        System Findings:
        - Deep Learning Label: {label}
        - Statistical Confidence: {confidence:.2f}%
        - Model Identity: {model_name}
        - Technical Summary: {technical_summary}
        
        Medical Context (Knowledge Base Reference):
        \"\"\"{kb_context}\"\"\"
        
        Your Task:
        1. Synthesize a professional clinical research report in English.
        2. Correlate the AI findings with the provided medical context (e.g., BI-RADS criteria).
        3. Discuss the relevance of the CNN+CLAHE approach for these specific markers.
        """
        
        response = model.generate_content(prompt)
        return {"agent_report": response.text}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
