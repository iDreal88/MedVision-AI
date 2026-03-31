# MedVision-AI Clinical Knowledge Base

## BI-RADS Assessment Criteria
- BI-RADS 1: Negative
- BI-RADS 2: Benign
- BI-RADS 3: Probably Benign
- BI-RADS 4: Suspicious (4A: Low, 4B: Moderate, 4C: High suspicion)
- BI-RADS 5: Highly Suggestive of Malignancy
- BI-RADS 6: Known Biopsy-Proven Malignancy

## Diagnostic Features
- **Masses:** Shape (Oval, Round, Irregular), Margin (Circumscribed, Obscured, Microlobulated, Ill-defined, Spiculated).
- **Calcifications:** Typically Benign vs. Suspicious Morphology (Amorphous, Coarsely Heterogeneous, Fine Pleomorphic).
- **Architecture:** Architectural distortion or focal asymmetry.

## AI Methodology
The MedVision-AI system utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance micro-calcifications before processing through a fine-tuned CNN architecture. This maximizes the detection of subtle architectural distortions.
