# Breast Cancer Pathology Knowledge Base

## Classification Overview
Breast lesions are generally classified into two categories:
1. **Benign**: Non-cancerous growths. They are typically well-defined, regular in shape, and do not invade surrounding tissues. common examples include cysts and fibroadenomas.
2. **Malignant**: Cancerous growths. They often have irregular borders, may show architectural distortion, and can invade nearby tissue or spread (metastasize).

## BI-RADS Assessment Categories
The Breast Imaging-Reporting and Data System (BI-RADS) is a standard for reporting:
- **BI-RADS 0**: Incomplete. Further imaging (e.g., spot compression, magnification, or ultrasound) is required.
- **BI-RADS 1**: Negative. Symmetrical breasts with no masses, architectural distortion, or suspicious calcifications.
- **BI-RADS 2**: Benign Findings. Includes secretory calcifications, simple cysts, and fat-containing lesions (hamartomas).
- **BI-RADS 3**: Probably Benign. <2% risk of malignancy. Recommend 6-month interval follow-up.
- **BI-RADS 4**: Suspicious. 2% to 95% likelihood of malignancy. Biopsy is typically indicated.
- **BI-RADS 5**: Highly Suggestive of Malignancy. >95% likelihood. Biopsy is mandatory.

## Radiographic Morphologies (Malignant)
- **Mass Shape**: Irregular shape is highly suspicious compared to round or oval shapes.
- **Margins**: Spiculated (star-shaped) margins have the highest positive predictive value for malignancy. Ill-defined or microlobulated margins also warrant suspicion.
- **Density**: Malignant masses are often more dense than surrounding parenchyma.
- **Calcification Patterns**: fine pleomorphic or fine linear/linear branching calcifications suggest Docctal Carcinoma In Situ (DCIS).

## Radiographic Morphologies (Benign)
- **Mass Shape**: Round or oval shapes are often associated with benignancy.
- **Margins**: Circumscribed (well-defined) margins are a hallmark of benign lesions like cysts or fibroadenomas.
- **Persistence**: Stable appearance over multiple years is a strong indicator of a benign condition.

## Explainable AI (Grad-CAM) Technical Interpretation
- **Activation Centroids**: In deep learning models, high-intensity Grad-CAM regions (the "red" areas) represent the spatial locations of the features that the neural network weighted most heavily during its final decision-making layer.
- **Feature Correlation**: In mammography models, these activations often align with areas of high optical density or architectural distortion.
- **Layer Specificity**: Using deep convolutional layers (like `conv5` in ResNet or `block5` in VGG) allows the Grad-CAM to capture complex structural patterns rather than just simple edges.

## Model-Specific Insights
- **CNN+CLAHE**: The use of Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances the visibility of micro-calcifications, allowing the CNN to focus on subtle high-frequency details.
- **ResNet50 / VGG**: Transfer learning models leverage pre-trained weights from ImageNet, making them highly sensitive to general object shapes and textures, which they adapt to recognize pathological breast tissue patterns.
