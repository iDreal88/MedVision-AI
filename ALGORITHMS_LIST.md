# List of Algorithms Used in Thesis

This document summarizes the machine learning and deep learning algorithms implemented and benchmarked in the research: **"REAL-TIME BREAST CANCER DIAGNOSIS SYSTEM USING CLAHE-ENHANCED DEEP LEARNING"**.

## 1. K-Nearest Neighbors (KNN) - Baseline
- **Description**: A classical non-parametric machine learning algorithm used for proximity-based classification.
- **Feature Extraction**: Uses Histogram of Oriented Gradients (HOG) to extract texture and geometric features from mammogram images.
- **Hyperparameters**: Optimized with $K=3$ using Euclidean distance.
- **Performance**: Achieved **95.30%** accuracy on the test dataset.

## 2. VGG16 (Convolutional Neural Network)
- **Description**: A deep CNN architecture consisting of 16 weight layers, known for its uniform architecture and deep feature extraction.
- **Implementation**: Utilized transfer learning with ImageNet weights and fine-tuned on the mammography dataset.
- **Layers**: Includes 13 convolutional layers and 3 fully connected layers.
- **Performance**: Achieved **97.51%** accuracy.

## 3. VGG19 (Convolutional Neural Network)
- **Description**: An extension of VGG16 with 19 weight layers (16 convolutional layers).
- **Implementation**: Employed to explore the impact of increased depth on morphological feature abstraction in medical imaging.
- **Performance**: Achieved **96.69%** accuracy.

## 4. CNN + CLAHE (Proposed Hybrid Pipeline)
- **Description**: The primary contribution of this thesis, combining localized contrast enhancement with a fine-tuned CNN (VGG16-based).
- **Preprocessing**: Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to mitigate noise and reveal micro-calcifications.
- **Architecture**: A custom sequential CNN architecture optimized for high-contrast grayscale input.
- **Performance**: Achieved the peak accuracy of **97.84%**.

## 5. ResNet50 (Residual Network)
- **Description**: A 50-layer deep network that utilizes "skip connections" or residual learning to overcome the vanishing gradient problem.
- **Implementation**: Used as a benchmark for deeper architectures in the cloud deployment environment.
- **Performance**: Achieved **95.01%** accuracy.

---

### Summary Table of Results

| Algorithm | Architecture Type | Accuracy (%) |
| :--- | :--- | :--- |
| **KNN** | Classical + HOG | 95.30% |
| **ResNet50** | Residual CNN | 95.01% |
| **VGG19** | Deep CNN | 96.69% |
| **VGG16** | Deep CNN | 97.51% |
| **CNN + CLAHE** | **Hybrid (Proposed)** | **97.84%** |

### Language Support
The MedVision-AI platform has been updated to support:
- English
- Chinese (Traditional)
- Bahasa Indonesia
- Japanese
- Korean

### RAG Engine (LLM)
The Retrieval-Augmented Generation (RAG) system utilizes the **Gemini-1.5-Flash** large language model as the generative backbone, combined with **Sentence-Transformers (all-MiniLM-L6-v2)** for semantic search across WHO-standard diagnostic guidelines.
