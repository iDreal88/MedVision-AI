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

## Pathological Correlation (ResNet50 Adaptation)
- **ResNet50 Feature Extraction**: Recent studies show that ResNet50 models are particularly effective at identifying subtle texture variations in mammograms, which may correlate with micro-calcifications.
- **Grad-CAM Insights**: Visualizing the Grad-CAM maps from the `conv5_block3_out` layer often pinpoints the exact centroid of suspected lesions.
