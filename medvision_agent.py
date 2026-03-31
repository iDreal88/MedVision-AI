import os
import json
import requests
import google.generativeai as genai

# Configuration
GEMINI_MODEL = "gemini-3-flash-preview" 
LOCAL_API_URL = "http://localhost:8000/predict"
KNOWLEDGE_BASE_FILE = "knowledge_base.md"

class MedVisionAgent:
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        
        # This orchestrated version avoids all heavy ML imports (Torch/TF) 
        # to ensure zero crashes on Mac M2.
        self.model = genai.GenerativeModel(model_name=GEMINI_MODEL)

    def run_diagnostic_research(self, image_path: str):
        """
        Orchestrates the analysis by calling the local Diagnostic Service 
        and synthesizing the findings with the provided medical knowledge base.
        """
        if not os.path.exists(image_path):
            return "Error: Image not found."

        print(f"1. Calling Diagnostic Service for {image_path}...")
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    LOCAL_API_URL, 
                    files={"file": f}, 
                    data={"model_name": "CNN+CLAHE"}
                )
            diag_data = response.json()
        except Exception as e:
            return f"Service Error: {e}"

        print(f"2. Reading Medical Knowledge Base ({KNOWLEDGE_BASE_FILE})...")
        kb_context = ""
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, "r") as f:
                kb_context = f.read()

        print("3. Synthesizing Final Clinical Report with Gemini 3 Flash...")
        prompt = f"""
        Role: Senior Resident Oncologist & AI Researcher.
        
        System Findings:
        - Deep Learning Label: {diag_data.get('label')}
        - Statistical Confidence: {diag_data.get('confidence'):.2f}%
        - Technical Summary: {diag_data.get('report')}
        
        Medical Context (Knowledge Base Reference):
        \"\"\"{kb_context}\"\"\"
        
        Your Task:
        1. Synthesize a professional clinical report in English.
        2. Correlate the AI findings with the provided medical context (e.g., BI-RADS criteria).
        3. Discuss the relevance of the CNN+CLAHE approach for these specific markers.
        """
        
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        agent = MedVisionAgent(api_key=api_key)
        test_img = "cnn+clahe/sample_test.jpg"
        if os.path.exists(test_img):
            print(f"MedVision-Agent (Gemini 3 / Stable Orchestrator) live.")
            print("-" * 50)
            result = agent.run_diagnostic_research(test_img)
            print(result)
        else:
            print("Missing test image.")
    else:
        print("Missing API key.")
