# MedVision AI - Technical Documentation

## 1. Project Overview
**MedVision AI** is an advanced medical research application designed to assist pathologists in diagnosing breast cancer from mammogram images. It utilizes an ensemble of Deep Learning models to predict malignancy and provides explainable results through visual heatmaps (Grad-CAM) and clinical reports generated with RAG (Retrieval-Augmented Generation).

---

## 2. Technology Stack

### **Frontend (User Interface)**
*   **Framework**: [React](https://react.dev/) (v18)
*   **Build Tool**: [Vite](https://vitejs.dev/)
*   **Styling**:
    *   **Tailwind CSS**: For responsive layout and utility-first styling.
    *   **Framer Motion**: For smooth, high-performance UI animations (tabs, page transitions).
    *   **Glassmorphism**: Custom CSS for the modern, translucent "glass" aesthetic.
*   **State Management**: React `useState` / `useEffect`.
*   **HTTP Client**: `Axios` for communicating with the Python backend.

### **Backend (AI Engine)**
*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
*   **Deep Learning**:
    *   **TensorFlow / Keras**: For loading and running the neural networks.
    *   **OpenCV (`cv2`)**: For image preprocessing (CLAHE, resizing).
    *   **PyTorch**: Used for specific RAG embeddings (via `sentence-transformers`).
*   **PDF Generation**: `fpdf2` for creating professional, structured PDF reports dynamically.
*   **RAG System**: Custom implementation using local knowledge retrieval from `knowledge_base.md`.

---

## 3. Hosting & Infrastructure

### **Frontend Deployment**
*   **Platform**: **Vercel**
*   **URL**: `https://medvisionai-project.vercel.app` (Example)
*   **Features Used**: Edge caching, continuous deployment from Git.

### **Backend Deployment**
*   **Platform**: **Hugging Face Spaces**
*   **SDK**: Docker
*   **Hardware**: 16GB RAM CPU Basic (Required for loading ResNet+VGG+CNN simultaneously).
*   **Container**: Custom `Dockerfile` based on `python:3.10-slim`.
*   **Reasoning**: Vercel/Netlify functions have a 50MB size limit and low RAM, which is insufficient for TensorFlow models. Hugging Face Spaces allows for larger containerized apps.

---

## 4. AI & Model Architecture

### **The "Ensemble" Approach**
The system does not rely on a single model. It uses three distinct architectures to provide a "Second Opinion" ecosystem:

1.  **CNN + CLAHE (The Specialist)**:
    *   **Architecture**: Custom Convolutional Neural Network.
    *   **Input**: Images preprocessed with **CLAHE** (Contrast Limited Adaptive Histogram Equalization).
    *   **Strength**: Extremely sensitive to structural anomalies; best at detecting masses in dense tissue.

2.  **ResNet50 (The Generalist)**:
    *   **Architecture**: Deep Residual Learning.
    *   **Strength**: High accuracy on standard datasets; robust against variations in image zoom/angle.

3.  **VGG16 / VGG19 (The Detail Finder)**:
    *   **Architecture**: Very deep networks with small receptive fields.
    *   **Strength**: Excellent at extracting granular textures, such as micro-calcifications.

### **Explainable AI (XAI)**
*   **Grad-CAM** (Gradient-weighted Class Activation Mapping):
    *   The system extracts gradients from the final convolutional layer to generate a "heatmap."
    *   **Red Areas**: Indicate regions the model acted upon (e.g., the tumor boundary).
    *   **Blue Areas**: Background tissue.
    *   *Purpose*: This proves the model isn't "cheating" (e.g., looking at watermarks) and is focusing on the actual lesion.

---

## 5. RAG (Retrieval-Augmented Generation)
Instead of a chatbot, the system uses a **safe, deterministic RAG pipeline**:

1.  **Diagnosis**: The model predicts "Cancer (98%)".
2.  **Retrieval**: The system searches `knowledge_base.md` for context related to "malignant mammogram features."
3.  **Generation**: It compiles a clinical report combining:
    *   The specific model confidence score.
    *   The Grad-CAM finding (e.g., "Focal high-intensity activation").
    *   Retrieved clinical guidelines (e.g., "Suggests BI-RADS 4/5").
4.  **Result**: Referenced, clinically grounded text without hallucinations.

---

## 6. Directory Structure
```
medvision-ai/
├── api/                   # Backend Code
│   ├── main.py            # FastAPI entry point & endpoints
│   ├── rag_engine.py      # Logic for Knowledge Base retrieval
│   └── knowledge_base.md  # Medical encyclopedia file
├── frontend/              # Frontend Code
│   └── src/
│       └── App.jsx        # Main React Logic
├── requirements.txt       # Python Dependencies
└── Dockerfile             # Server configuration for Hugging Face
```
