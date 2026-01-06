# Breast Cancer Pathology Knowledge Base

## Classification Overview
Breast lesions are generally classified into two categories:
1. **Benign**: Non-cancerous growths. They are typically well-defined, regular in shape, and do not invade surrounding tissues.
2. **Malignant**: Cancerous growths. They often have irregular borders, may show architectural distortion, and can invade nearby tissue or spread (metastasize).

## Radiographic Features (Mammography)
- **Masses**: Evaluated based on shape (round, oval, irregular) and margins (circumscribed, obscured, microlobulated, ill-defined, speculated).
- **Calcifications**: Small deposits of calcium. Some patterns (pleomorphic, linear branching) are highly suspicious for malignancy.
- **Architectural Distortion**: The normal anatomy is pushed or pulled, which can be a sign of underlying malignancy.

## Explainable AI (Grad-CAM) Interpretation
- **High Heatmap Intensity (Red)**: Indicates the regions that most strongly influenced the model's prediction.
- **Benign Patterns**: Focus on smooth, regular structures or areas with no significant density.
- **Malignant Patterns**: Focus on dense, irregular masses or areas with speculated margins.

## Diagnostic Recommendations
- **BI-RADS 0**: Incomplete, needs further imaging.
- **BI-RADS 1-2**: Negative or Benign findings. Routine screening recommended.
- **BI-RADS 3**: Probably benign. Short-interval follow-up recommended.
- **BI-RADS 4-5**: Suspicious or Highly Suggestive of malignancy. Biopsy is typically recommended.

## Pathological Correlation (VGG16)
- **VGG16 Feature Extraction**: VGG16 models are robust at capturing macro-textures and irregular margins commonly found in mammographic lesions.
- **Block5_conv3 Activation**: The final convolutional layer in VGG16 provides a coarse but reliable localizing signal for dense masses.
