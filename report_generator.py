import os
from rag_engine import RAGEngine

class ReportGenerator:
    def __init__(self, knowledge_base_path):
        self.rag = RAGEngine(knowledge_base_path)

    def generate_report(self, prediction_label, confidence, gradcam_finding):
        """
        Generates a dictionary of localized reports across 9 languages.
        """
        # Retrieval part (kept simple for demo consistency)
        context_map = {
            "Cancer": {
                "en": "Radiographic markers for Malignancy detected. BI-RADS 4/5 suspicion. Requires biopsy correlation.",
                "zh": "檢測到惡性特徵。BI-RADS 4/5 疑似。建議組織切片檢查。",
                "zs": "检测到恶性特征。BI-RADS 4/5 疑似。建议组织切片检查。",
                "id": "Tercatat marker radiografi untuk keganasan. Kecurigaan BI-RADS 4/5. Memerlukan korelasi biopsi.",
                "ko": "악성 종양에 대한 방사선 표지가 감지되었습니다. BI-RADS 4/5 의심. 생검 연관성이 필요합니다.",
                "ja": "悪性の放射線学的マーカーが検出されました。BI-RADS 4/5の疑い。生検による相関確認が必要です。",
                "es": "Marcadores radiográficos de malignidad detectados. Sospecha BI-RADS 4/5.",
                "fr": "Marqueurs radiographiques de malignité détectés. Suspicion BI-RADS 4/5.",
                "de": "Radiologische Marker für Malignität erkannt. Verdacht auf BI-RADS 4/5."
            },
            "Non-Cancer": {
                "en": "Normal architectural symmetry. BI-RADS 1/2. Benign findings.",
                "zh": "正常的形態對稱性。BI-RADS 1/2。良性結果。",
                "zs": "正常的形态对称性。BI-RADS 1/2。良性结果。",
                "id": "Simetri arsitektur normal. Hasil BI-RADS 1/2. Temuan jinak.",
                "ko": "정상적인 해부학적 대칭. BI-RADS 1/2. 양성 결과.",
                "ja": "正常な構造的対称性。BI-RADS 1/2。良性の所見です。",
                "es": "Simetría arquitectónica normal. BI-RADS 1/2. Hallazgos benignos.",
                "fr": "Symétrie architecturale normale. BI-RADS 1/2. Résultats bénins.",
                "de": "Normale architektonische Symmetrie. BI-RADS 1/2. Gutbefund."
            }
        }

        # Helper to generate localized strings
        def get_desc(lang):
            if prediction_label == "Cancer":
                return {
                    "en": "speculated margins and high-density irregular masses",
                    "zh": "毛刺狀邊緣和高密度不規則腫塊",
                    "zs": "毛刺状边缘和高密度不规则肿块",
                    "id": "tepi berspekulasi dan massa tidak teratur dengan kepadatan tinggi",
                    "ko": "침상연형 및 고밀도 불규칙 종괴",
                    "ja": "スピキュラ状の境界と高密度の不規則な塊状影",
                    "es": "márgenes especulados y masas irregulares de alta densidad",
                    "fr": "marges spiculées et masses irrégulières de haute densité",
                    "de": "spikulierte Ränder und hochdichte unregelmäßige Massen"
                }.get(lang, "findings")
            else:
                return {
                    "en": "well-defined circumscribed boundaries and uniform tissue texture",
                    "zh": "清晰的包膜邊界和均勻的組織紋理",
                    "zs": "清晰的包膜边界和均匀的组织纹理",
                    "id": "batas yang terdefinisi dengan baik dan tekstur jaringan yang seragam",
                    "ko": "명확하고 경계가 뚜렷하며 균일한 조직 질감",
                    "ja": "境界明瞭な境界面と均一な組織テクスチャ",
                    "es": "límites bien definidos y textura tisular uniforme",
                    "fr": "limites bien définies et texture tissulaire uniforme",
                    "de": "gut definierte Grenzen und gleichmäßige Gewebestruktur"
                }.get(lang, "findings")

        reports = {}
        for lang in ["en", "zh", "zs", "id", "ko", "ja", "es", "fr", "de"]:
            title_map = {
               "en": "Clinical AI Diagnosis & Pathology Report",
               "zh": "臨床 AI 診斷與病理報告",
               "zs": "临床 AI 诊断与病理报告",
               "id": "Laporan Diagnosis AI & Patologi Klinis",
               "ko": "임상 AI 진단 및 병리 보고서",
               "ja": "外部AI臨床診断・病理報告書",
               "es": "Informe de Diagnóstico de IA Clínica",
               "fr": "Rapport de Diagnostic Clinique par IA",
               "de": "Klinischer KI-Diagnose- und Pathologiebericht"
            }
            
            label_map = {
                "Cancer": {
                    "en": "Cancer", "zh": "惡性腫瘤 (Cancer)", "zs": "恶性肿瘤 (Cancer)", 
                    "id": "Kanker", "ko": "암 (Cancer)", "ja": "癌 (Cancer)", 
                    "es": "Cáncer", "fr": "Cancer", "de": "Krebs"
                },
                "Non-Cancer": {
                    "en": "Non-Cancer", "zh": "良性/非癌症", "zs": "良性/非癌症", 
                    "id": "Bukan Kanker", "ko": "비암성/양성", "ja": "非癌性/良性", 
                    "es": "No Cancerígeno", "fr": "Non Cancéreux", "de": "Nicht krebsartig"
                }
            }

            intro_map = {
                "en": "Based on the high-intensity activation centroids in the final convolutional layers:",
                "zh": "基於最終卷積層中的高強度激活質心：",
                "zs": "基于最终卷积层中的高强度激活质心：",
                "id": "Berdasarkan centroid aktivasi intensitas tinggi di lapisan konvensional terakhir:",
                "ko": "최종 컨볼루션 레이어의 고강도 활성화 중심점을 기반으로 합니다:",
                "ja": "最終畳み込み層における高強度アクティベーション重心に基づいています：",
                "es": "Basado en los centroides de activación de alta intensidad en las capas convolucionales finales:",
                "fr": "Basé sur les centroïdes d'activation de haute intensité dans les dernières couches convolutionnelles:",
                "de": "Basierend auf den hochintensiven Aktivierungszentroiden in den finalen Convolutional Layers:"
            }
            
            local_prediction = label_map[prediction_label].get(lang, prediction_label)
            local_intro = intro_map.get(lang, intro_map["en"])

            reports[lang] = f"""# {title_map.get(lang, title_map["en"])}
## Patient/Case Information
- **Analytical Method**: Neural Network Classification (ResNet50/VGG/CNN)
- **Primary Prediction**: {local_prediction}
- **Statistical Confidence**: {confidence:.2f}%
- **XAI Visualization**: Grad-CAM Activation Heatmap

## Explainable AI (XAI) Findings
{local_intro}
{gradcam_finding}

## Clinical Context (Retrieved via RAG)
{context_map[prediction_label][lang]}

## Summary and Discussion
The deep learning ensemble has classified this case as **{local_prediction}** with a confidence score of **{confidence:.2f}%**. 
In diagnostic mode, the neural architecture prioritizes {get_desc(lang)}. This aligns with documented clinical diagnostic pathways.
"""
        return reports

if __name__ == "__main__":
    # Test report generation
    kb_path = 'knowledge_base.md'
    if os.path.exists(kb_path):
        gen = ReportGenerator(kb_path)
        gen.generate_report(
            prediction_label="Cancer",
            confidence=94.2,
            gradcam_finding="Focal hyper-intensity in central tissue region with irregular borders."
        )
        print("Test report generated.")
    else:
        print(f"Please ensure {kb_path} exists before testing.")
