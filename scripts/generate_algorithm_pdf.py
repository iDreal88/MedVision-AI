from fpdf import FPDF
import os

class AlgorithmPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59) # Slate 800
        self.rect(0, 0, 210, 40, 'F')
        self.set_font("helvetica", "B", 20)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 10)
        self.cell(0, 15, "MedVision-AI: Thesis Algorithm Summary", ln=True)
        self.set_font("helvetica", "I", 10)
        self.cell(0, 5, "Technical Appendix for Master's Thesis", ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align='C')

def create_algorithm_list_pdf():
    pdf = AlgorithmPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. KNN
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(37, 99, 235) # Blue 600
    pdf.cell(0, 10, "1. K-Nearest Neighbors (KNN)", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, "- Description: Classical non-parametric method using HOG features.\n"
                         "- Best K Value: 3 (Euclidean Distance)\n"
                         "- Accuracy: 95.30%")
    pdf.ln(5)

    # 2. VGG16
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 10, "2. VGG16 (CNN)", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, "- Description: 16-layer Deep Convolutional Neural Network.\n"
                         "- Method: Transfer Learning with ImageNet weights.\n"
                         "- Accuracy: 97.51%")
    pdf.ln(5)

    # 3. VGG19
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 10, "3. VGG19 (CNN)", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, "- Description: Deeper 19-layer architectural branch.\n"
                         "- Accuracy: 96.69%")
    pdf.ln(5)

    # 4. ResNet50
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 10, "4. ResNet50 (Residual Network)", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, "- Description: 50-layer deep network with skip connections.\n"
                         "- Accuracy: 95.01%")
    pdf.ln(5)

    # 5. CNN + CLAHE
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(16, 185, 129) # Emerald 500
    pdf.cell(0, 10, "5. CNN + CLAHE (Proposed Optimized Pipeline)", ln=True)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(5, 150, 105)
    pdf.multi_cell(0, 6, "- Innovation: Localized contrast enhancement for micro-calcification detection.\n"
                         "- Peak Accuracy: 97.84%")
    pdf.ln(10)

    # Summary Table
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(100, 10, "Algorithm Name", border=1, fill=True)
    pdf.cell(40, 10, "Accuracy (%)", border=1, fill=True)
    pdf.ln()

    data = [
        ("KNN", "95.30%"),
        ("ResNet50", "95.01%"),
        ("VGG19", "96.69%"),
        ("VGG16", "97.51%"),
        ("CNN+CLAHE", "97.84%"),
    ]

    pdf.set_font("helvetica", "", 11)
    for name, acc in data:
        pdf.cell(100, 10, name, border=1)
        pdf.cell(40, 10, acc, border=1)
        pdf.ln()

    # RAG Section
    pdf.ln(10)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "RAG Engine Technology Stack", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, "- LLM Core: Gemini-1.5-Flash\n"
                         "- Semantic Search: Sentence-Transformers (all-MiniLM-L6-v2)\n"
                         "- Database: FAISS-indexed WHO Diagnostic Criteria.")

    # Multi-language
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Supported Languages (i18n)", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, "- English, Traditional Chinese, Bahasa Indonesia, Japanese, Korean.")

    pdf.output("thesis_algorithm_summary.pdf")
    print("PDF generated successfully: thesis_algorithm_summary.pdf")

if __name__ == "__main__":
    create_algorithm_list_pdf()
