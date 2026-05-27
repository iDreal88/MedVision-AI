import os
from .rag_engine import RAGEngine

class ReportGenerator:
    def __init__(self, knowledge_base_path):
        self.rag = RAGEngine(knowledge_base_path)

    def generate_report(self, prediction_label, confidence, gradcam_finding, output_path=None, language="en"):
        """
        Generates a structured pathology report based on model output and RAG.
        """
        # 1. Retrieve clinical context based on the prediction and findings
        query = f"Radiographic markers and BI-RADS for {prediction_label} breast cancer"
        context = self.rag.search(query, top_k=3)

        # Translations Dictionary
        T = {
            "en": {
                "title": "Clinical AI Diagnosis & Pathology Report",
                "patient_info": "Patient/Case Information",
                "method": "Analytical Method",
                "primary_pred": "Primary Prediction",
                "confidence": "Statistical Confidence",
                "xai_viz": "XAI Visualization",
                "xai_findings": "Explainable AI (XAI) Findings",
                "xai_desc": "Based on the high-intensity activation centroids in the final convolutional layers:",
                "clinical_context": "Clinical Context (Retrieved via RAG)",
                "clinical_desc": f"The following insights were retrieved from the medical knowledge base based on the suspicion of {prediction_label}:",
                "summary_title": "Summary and Discussion",
                "summary_1": f"The deep learning ensemble has classified this case as **{prediction_label}** with a confidence score of **{confidence:.2f}%**.",
                "summary_2_cancer": "In cancer cases, the model typically prioritizes spiculated margins and high-density irregular masses.",
                "summary_2_benign": "In non-cancer cases, the model typically prioritizes well-defined circumscribed boundaries and uniform tissue texture.",
                "summary_3_cancer": "The retrieved clinical context suggests that a BI-RADS 4 or 5 assessment might be considered, warranting further pathological correlation or biopsy.",
                "summary_3_benign": "The retrieved clinical context suggests that this presentation aligns with BI-RADS 2 or 3, suggesting benign findings or a low-risk probably benign condition."
            },
            "zh": {
                "title": "臨床 AI 診斷與病理報告",
                "patient_info": "病患/病例資訊",
                "method": "分析方法",
                "primary_pred": "主要預測",
                "confidence": "統計置信度",
                "xai_viz": "可解釋 AI (XAI) 視覺化",
                "xai_findings": "可解釋 AI (XAI) 發現",
                "xai_desc": "基於最終卷積層中的高強度激活質心：",
                "clinical_context": "臨床背景 (透過 RAG 檢索)",
                "clinical_desc": f"基於對 {prediction_label} 的懷疑，從醫學知識庫中檢索到以下見解：",
                "summary_title": "總結與討論",
                "summary_1": f"深度學習集成模型將此病例分類為 **{prediction_label}**，置信度為 **{confidence:.2f}%**。",
                "summary_2_cancer": "在癌症病例中，模型通常會優先考慮毛刺狀邊緣和高密度不規則腫塊。",
                "summary_2_benign": "在非癌症病例中，模型通常會優先考慮輪廓清晰的邊界和均勻的組織紋理。",
                "summary_3_cancer": "檢索到的臨床背景表明，可以考慮 BI-RADS 4 或 5 評估，需要進一步的病理相關性分析或活檢。",
                "summary_3_benign": "檢索到的臨床背景表明，此表現符合 BI-RADS 2 或 3，暗示為良性發現或低風險的可能良性狀況。"
            },
            "zs": {
                "title": "临床 AI 诊断与病理报告",
                "patient_info": "病患/病例资讯",
                "method": "分析方法",
                "primary_pred": "主要预测",
                "confidence": "统计置信度",
                "xai_viz": "可解释 AI (XAI) 视觉化",
                "xai_findings": "可解释 AI (XAI) 发现",
                "xai_desc": "基于最终卷积层中的高强度激活质心：",
                "clinical_context": "临床背景 (透过 RAG 检索)",
                "clinical_desc": f"基于对 {prediction_label} 的怀疑，从医学知识库中检索到以下见解：",
                "summary_title": "总结与讨论",
                "summary_1": f"深度学习集成模型将此病例分类为 **{prediction_label}**，置信度为 **{confidence:.2f}%**。",
                "summary_2_cancer": "在癌症病例中，模型通常会优先考虑毛刺状边缘和高密度不规则肿块。",
                "summary_2_benign": "在非癌症病例中，模型通常会优先考虑轮廓清晰的边界和均匀的组织纹理。",
                "summary_3_cancer": "检索到的临床背景表明，可以考虑 BI-RADS 4 或 5 评估，需要进一步的病理相关性分析或活检。",
                "summary_3_benign": "检索到的临床背景表明，此表现符合 BI-RADS 2 或 3，暗示为良性发现或低风险的可能良性状况。"
            },
            "ja": {
                "title": "臨床 AI 診断および病理レポート",
                "patient_info": "患者/症例情報",
                "method": "分析方法",
                "primary_pred": "主要予測",
                "confidence": "統計的信頼度",
                "xai_viz": "XAI 視覚化",
                "xai_findings": "説明可能な AI (XAI) の所見",
                "xai_desc": "最終畳み込み層における高強度活性化セントロイドに基づく：",
                "clinical_context": "臨床的背景 (RAG 経由で検索)",
                "clinical_desc": f"{prediction_label} の疑いに基づき、医学知識ベースから以下の知見が検索されました：",
                "summary_title": "要約と考察",
                "summary_1": f"深層学習アンサンブルは、この症例を信頼度スコア **{confidence:.2f}%** で **{prediction_label}** に分類しました。",
                "summary_2_cancer": "がんの症例において、モデルは通常、スピキュラを伴う辺縁や高密度の不規則な腫瘤を重視します。",
                "summary_2_benign": "非がんの症例において、モデルは通常、境界明瞭な辺縁や均一な組織テクスチャを重視します。",
                "summary_3_cancer": "検索された臨床的背景は、BI-RADS 4 または 5 の評価が検討される可能性があり、さらなる病理学的相関関係または生検が必要であることを示唆しています。",
                "summary_3_benign": "検索された臨床的背景は、この所見が BI-RADS 2 または 3 と一致しており、良性の所見または低リスクの良性の可能性が高い状態を示唆しています。"
            },
            "id": {
                "title": "Laporan Diagnosis AI Klinis & Patologi",
                "patient_info": "Informasi Pasien/Kasus",
                "method": "Metode Analitis",
                "primary_pred": "Prediksi Utama",
                "confidence": "Tingkat Kepercayaan",
                "xai_viz": "Visualisasi XAI",
                "xai_findings": "Temuan Explainable AI (XAI)",
                "xai_desc": "Berdasarkan sentroid aktivasi intensitas tinggi di lapisan konvolusional akhir:",
                "clinical_context": "Konteks Klinis (Diambil via RAG)",
                "clinical_desc": f"Wawasan berikut diambil dari basis pengetahuan medis berdasarkan dugaan {prediction_label}:",
                "summary_title": "Ringkasan dan Diskusi",
                "summary_1": f"Ansambel pembelajaran mendalam telah mengklasifikasikan kasus ini sebagai **{prediction_label}** dengan tingkat kepercayaan **{confidence:.2f}%**.",
                "summary_2_cancer": "Pada kasus kanker, model biasanya memprioritaskan margin berbentuk bintang (spiculated) dan massa ireguler dengan kepadatan tinggi.",
                "summary_2_benign": "Pada kasus non-kanker, model biasanya memprioritaskan batas tegas yang sirkumskrip dan tekstur jaringan yang seragam.",
                "summary_3_cancer": "Konteks klinis yang diambil menunjukkan bahwa penilaian BI-RADS 4 atau 5 mungkin dipertimbangkan, menjamin korelasi patologis lebih lanjut atau biopsi.",
                "summary_3_benign": "Konteks klinis yang diambil menunjukkan bahwa presentasi ini sejalan dengan BI-RADS 2 atau 3, menunjukkan temuan jinak atau kondisi yang kemungkinan jinak dengan risiko rendah."
            },
            "ko": {
                "title": "임상 AI 진단 및 병리 보고서",
                "patient_info": "환자/증례 정보",
                "method": "분석 방법",
                "primary_pred": "주요 예측",
                "confidence": "통계적 신뢰도",
                "xai_viz": "XAI 시각화",
                "xai_findings": "설명 가능한 AI (XAI) 결과",
                "xai_desc": "최종 컨볼루션 레이어의 고강도 활성화 중심을 기반으로 함:",
                "clinical_context": "임상 배경 (RAG를 통해 검색됨)",
                "clinical_desc": f"{prediction_label} 의심을 바탕으로 의학 지식 베이스에서 다음 정보를 검색했습니다:",
                "summary_title": "요약 및 논의",
                "summary_1": f"딥러닝 앙상블은 이 증례를 **{confidence:.2f}%**의 신뢰도로 **{prediction_label}**(으)로 분류했습니다.",
                "summary_2_cancer": "암 증례의 경우, 모델은 일반적으로 침상 가장자리 및 고밀도 불규칙한 종괴를 우선시합니다.",
                "summary_2_benign": "비암 증례의 경우, 모델은 일반적으로 윤곽이 뚜렷한 경계 및 균일한 조직 질감을 우선시합니다.",
                "summary_3_cancer": "검색된 임상 배경은 BI-RADS 4 또는 5 평가가 고려될 수 있으며 추가적인 병리학적 상관관계 확인이나 생검이 필요함을 시사합니다.",
                "summary_3_benign": "검색된 임상 배경은 이 소견이 BI-RADS 2 또는 3과 일치하며, 양성 소견이나 저위험군의 양성 가능성 질환을 시사함을 나타냅니다."
            },
            "es": {
                "title": "Diagnóstico Clínico por IA e Informe de Patología",
                "patient_info": "Información del Paciente/Caso",
                "method": "Método Analítico",
                "primary_pred": "Predicción Principal",
                "confidence": "Confianza Estadística",
                "xai_viz": "Visualización XAI",
                "xai_findings": "Hallazgos de IA Explicable (XAI)",
                "xai_desc": "Basado en los centroides de activación de alta intensidad en las últimas capas convolucionales:",
                "clinical_context": "Contexto Clínico (Recuperado vía RAG)",
                "clinical_desc": f"Se recuperaron los siguientes datos de la base de conocimientos médicos en base a la sospecha de {prediction_label}:",
                "summary_title": "Resumen y Discusión",
                "summary_1": f"El conjunto de aprendizaje profundo ha clasificado este caso como **{prediction_label}** con una confianza de **{confidence:.2f}%**.",
                "summary_2_cancer": "En casos de cáncer, el modelo típicamente prioriza márgenes espiculados y masas irregulares de alta densidad.",
                "summary_2_benign": "En casos sin cáncer, el modelo típicamente prioriza límites bien definidos circunscritos y textura de tejido uniforme.",
                "summary_3_cancer": "El contexto clínico recuperado sugiere que podría considerarse una evaluación BI-RADS 4 o 5, lo que justifica una mayor correlación patológica o biopsia.",
                "summary_3_benign": "El contexto clínico recuperado sugiere que esta presentación concuerda con BI-RADS 2 o 3, sugiriendo hallazgos benignos o una condición probablemente benigna de bajo riesgo."
            },
            "fr": {
                "title": "Diagnostic Clinique par IA et Rapport de Pathologie",
                "patient_info": "Informations Patient/Cas",
                "method": "Méthode Analytique",
                "primary_pred": "Prédiction Principale",
                "confidence": "Confiance Statistique",
                "xai_viz": "Visualisation XAI",
                "xai_findings": "Résultats de l'IA Explicable (XAI)",
                "xai_desc": "Basé sur les centroïdes d'activation à haute intensité dans les dernières couches convolutives:",
                "clinical_context": "Contexte Clinique (Récupéré via RAG)",
                "clinical_desc": f"Les informations suivantes ont été récupérées dans la base de connaissances médicales sur la base d'une suspicion de {prediction_label}:",
                "summary_title": "Résumé et Discussion",
                "summary_1": f"L'ensemble d'apprentissage profond a classé ce cas comme **{prediction_label}** avec une confiance de **{confidence:.2f}%**.",
                "summary_2_cancer": "Dans les cas de cancer, le modèle donne généralement la priorité aux marges spiculées et aux masses irrégulières de haute densité.",
                "summary_2_benign": "Dans les cas non cancéreux, le modèle donne généralement la priorité à des limites circonscrites bien définies et à une texture tissulaire uniforme.",
                "summary_3_cancer": "Le contexte clinique récupéré suggère qu'une évaluation BI-RADS 4 ou 5 pourrait être envisagée, justifiant une corrélation pathologique plus poussée ou une biopsie.",
                "summary_3_benign": "Le contexte clinique récupéré suggère que cette présentation correspond à un BI-RADS 2 ou 3, suggérant des résultats bénins ou une condition probablement bénigne à faible risque."
            },
            "de": {
                "title": "Klinischer KI-Diagnose- und Pathologiebericht",
                "patient_info": "Patienten-/Fallinformationen",
                "method": "Analytische Methode",
                "primary_pred": "Hauptvorhersage",
                "confidence": "Statistische Konfidenz",
                "xai_viz": "XAI-Visualisierung",
                "xai_findings": "Ergebnisse der Erklärbaren KI (XAI)",
                "xai_desc": "Basierend auf den hochintensiven Aktivierungs-Zentroiden in den finalen Faltungsschichten:",
                "clinical_context": "Klinischer Kontext (via RAG abgerufen)",
                "clinical_desc": f"Die folgenden Erkenntnisse wurden basierend auf dem Verdacht auf {prediction_label} aus der medizinischen Wissensdatenbank abgerufen:",
                "summary_title": "Zusammenfassung und Diskussion",
                "summary_1": f"Das Deep-Learning-Ensemble hat diesen Fall mit einer Konfidenz von **{confidence:.2f}%** als **{prediction_label}** klassifiziert.",
                "summary_2_cancer": "In Krebsfällen priorisiert das Modell typischerweise spikulierte Ränder und hochdichte unregelmäßige Massen.",
                "summary_2_benign": "In nicht-krebsartigen Fällen priorisiert das Modell typischerweise gut definierte, umschriebene Grenzen und eine gleichmäßige Gewebestruktur.",
                "summary_3_cancer": "Der abgerufene klinische Kontext deutet darauf hin, dass eine BI-RADS 4 oder 5 Bewertung in Betracht gezogen werden könnte, was eine weitere pathologische Korrelation oder Biopsie rechtfertigt.",
                "summary_3_benign": "Der abgerufene klinische Kontext deutet darauf hin, dass diese Präsentation mit BI-RADS 2 oder 3 übereinstimmt, was auf gutartige Befunde oder eine risikoarme, wahrscheinlich gutartige Erkrankung hindeutet."
            }
        }
        
        # Fallback to English if language not found
        lang_dict = T.get(language, T["en"])
        
        # Select summary texts
        summary_2 = lang_dict["summary_2_cancer"] if prediction_label == "Cancer" else lang_dict["summary_2_benign"]
        summary_3 = lang_dict["summary_3_cancer"] if prediction_label == "Cancer" else lang_dict["summary_3_benign"]

        # 2. Construct the report with localized text
        report_template = f'''# {lang_dict["title"]}

## {lang_dict["patient_info"]}
- **{lang_dict["method"]}**: Neural Network Classification (ResNet50/VGG/CNN)
- **{lang_dict["primary_pred"]}**: {prediction_label}
- **{lang_dict["confidence"]}**: {confidence:.2f}%
- **{lang_dict["xai_viz"]}**: Grad-CAM Activation Heatmap

## {lang_dict["xai_findings"]}
{lang_dict["xai_desc"]}
{gradcam_finding}

## {lang_dict["clinical_context"]}
{lang_dict["clinical_desc"]}
{context}

## {lang_dict["summary_title"]}
{lang_dict["summary_1"]} 

{summary_2} 

{summary_3}
'''
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_template)
            
        return report_template

if __name__ == "__main__":
    # Test report generation
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'knowledge_base.md')
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
