import os
from rag_engine import RAGEngine

class ReportGenerator:
    def __init__(self, knowledge_base_path):
        self.rag = RAGEngine(knowledge_base_path)

    def generate_report(self, prediction_label, confidence, gradcam_finding, output_path=None):
        """
        Generates a structured pathology report based on model output and RAG.
        """
        # 1. Retrieve clinical context based on the prediction and findings
        query = f"Radiographic markers and BI-RADS for {prediction_label} breast cancer"
        context = self.rag.search(query, top_k=3)

        # 2. Construct the report with a more clinical structure
        report_template = f"""# Clinical AI Diagnosis & Pathology Report

## Patient/Case Information
- **Analytical Method**: Neural Network Classification (ResNet50/VGG/CNN)
- **Primary Prediction**: {prediction_label}
- **Statistical Confidence**: {confidence:.2f}%
- **XAI Visualization**: Grad-CAM Activation Heatmap

## Explainable AI (XAI) Findings
Based on the high-intensity activation centroids in the final convolutional layers:
{gradcam_finding}

## Clinical Context (Retrieved via RAG)
The following insights were retrieved from the medical knowledge base based on the suspicion of {prediction_label}:
{context}

## Summary and Discussion
The deep learning ensemble has classified this case as **{prediction_label}** with a confidence score of **{confidence:.2f}%**. 

In {prediction_label.lower()} cases, the model typically prioritizes { "speculated margins and high-density irregular masses" if prediction_label == "Cancer" else "well-defined circumscribed boundaries and uniform tissue texture" }. 

The retrieved clinical context suggests that { "a BI-RADS 4 or 5 assessment might be considered, warranting further pathological correlation or biopsy" if prediction_label == "Cancer" else "this presentation aligns with BI-RADS 2 or 3, suggesting benign findings or a low-risk probably benign condition" }.

"""
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_template)
            
        return report_template

if __name__ == "__main__":
    # Test report generation
    kb_path = 'knowledge_base.md'
    if os.path.exists(kb_path):
        gen = ReportGenerator(kb_path)
        gen.generate_report(
            prediction_label="Cancer",
            confidence=94.2,
            gradcam_finding="Focal hyper-intensity in central tissue region with irregular borders.",
            output_path="output/reports/test_report.md"
        )
        print("Test report generated at output/reports/test_report.md")
    else:
        print(f"Please ensure {kb_path} exists before testing.")
