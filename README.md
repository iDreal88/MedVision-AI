---
title: MedVision AI
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# MedVision-AI: Real-Time Breast Cancer Diagnosis System 🏥

**MedVision-AI** is a comprehensive, cloud-integrated diagnostic framework designed to classify breast malignancies using a hybrid approach that combines advanced deep learning (CNNs) with strategic image optimization (CLAHE). Developed as a Master's Thesis project (*Real-Time Breast Cancer Diagnosis System Using CLAHE-Enhanced Deep Learning and Cloud Deployment*), this platform bridges the gap between laboratory research and accessible clinical tools.

## 🌟 Key Features

*   **Multi-Model Architecture:** Run real-time inference using fine-tuned architectures including an optimized **CNN+CLAHE pipeline (97.84% accuracy)**, VGG16, VGG19, ResNet50, and a baseline K-Nearest Neighbors (KNN) model.
*   **Explainable AI (XAI):** Addresses the "Black Box" problem by utilizing **Gradient-weighted Class Activation Mapping (Grad-CAM)**. The system generates high-resolution heatmaps overlaid on the original mammograms, providing clinicians with visual evidence of localized malignant clusters (micro-calcifications and architectural distortions).
*   **RAG-Enhanced Clinical Reporting:** Integrates a **Retrieval-Augmented Generation (RAG)** engine. Instead of providing binary confidence scores, the backend synthesizes model outputs with established WHO clinical guidelines to generate automated, medically compliant pathology reports (exportable as PDF).
*   **Production-Ready Cloud Deployment:** Features a responsive, glassmorphic UI built in React 18, communicating with a containerized FastAPI Python backend to deliver "Inference-as-a-Service" with low latency (average ~450ms).
*   **Persistent Analysis Log:** Maintains session history using `localStorage`, allowing clinicians to review and compare historical diagnostic records, complete with visual evidence and statistical confidence metrics.

## 📊 Model Performance Benchmarks

The deep learning models were trained on a robust dataset of over **10,430 biopsy-verified mammograms** (DDSM/Mendeley), enhanced via Contrast Limited Adaptive Histogram Equalization (CLAHE) and data augmentation.

| Model Architecture | Average Inference Latency (ms) | Peak Classification Accuracy (%) |
| :--- | :--- | :--- |
| **CNN + CLAHE (Hybrid)** | - | **97.84%** |
| VGG16 (Optimized) | 450ms | 97.51% |
| VGG19 | 510ms | 96.69% |
| ResNet50 | 580ms | 95.01% |
| KNN (Baseline) | 120ms | 95.30% |

> *Note: The VGG16+CLAHE pipeline achieves an Area Under the Curve (AUC) of 0.991, demonstrating near-perfect diagnostic capability in distinguishing malignant lesions from dense glandular tissue.*

## 🛠️ Technology Stack

*   **Frontend UI:** React 18, Vite, Framer Motion (Animations), Tailwind CSS, Lucide Icons.
*   **AI Backend:** FastAPI (Python), Uvicorn.
*   **Neural Engine & XAI:** TensorFlow 2.15+, Keras, OpenCV (CLAHE Image Processing).
*   **Knowledge Retrieval (RAG):** FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`).
*   **Infrastructure:** Docker, Google Antigravity IDE.

## 🚀 Getting Started

Follow these steps to set up and run the MedVision-AI platform on your local machine.

### Prerequisites

*   **Python:** 3.10 or 3.11 (Recommended)
*   **Node.js:** v18 or higher
*   **npm:** v9 or higher

### 1. Backend Setup (FastAPI + Neural Engine)

The backend handles image preprocessing, neural network inference (VGG/ResNet), Grad-CAM generation, and RAG-based reporting.

1.  **Navigate to the backend directory:**
    ```bash
    cd api
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On macOS/Linux:
    python -m venv .venv
    source .venv/bin/activate

    # On Windows:
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r ../requirements.txt
    ```

4.  **Verify Model Files:**
    Ensure the following `.keras` files are present in their respective directories:
    - `models/cnn_clahe/model_cnn_clahe.keras`
    - `models/resnet50/model_resnet50.keras`
    - `models/vgg16/model_vgg16.keras`
    - `models/vgg19/model_vgg19.keras`

5.  **Start the API server:**
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    *The API will be available at `http://localhost:8000`. You can access the interactive documentation at `http://localhost:8000/docs`.*

### 2. Frontend Setup (React + Vite)

The frontend provides the clinical dashboard, real-time feedback, and interactive XAI heatmaps.

1.  **Open a new terminal and navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the `frontend` folder:
    ```env
    VITE_API_URL=http://localhost:8000
    ```

4.  **Run the development server:**
    ```bash
    npm run dev
    ```
    *The UI will be accessible at `http://localhost:5173`.*

## 📂 Project Structure

*   **/api**: FastAPI scripts and backend logic.
*   **/frontend**: React source code and UI components.
*   **/models**: Organized directories containing pre-trained model weights and scripts.
*   **/notebooks**: Experimental analysis and visualization code.
*   **rag_engine.py**: Core logic for the Retrieval-Augmented Generation system.
*   **report_generator.py**: Module for synthesizing medical pathology reports.

## ⚖️ Disclaimer

*MedVision-AI is developed for research and educational purposes. The AI outcomes and reports should always be validated by a licensed medical professional before clinical application.*
