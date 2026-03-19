# MedVision AI - 技術文件

## 1. 專案概覽
**MedVision AI** 是一項先進的醫學研究應用程式，旨在協助病理學家從乳房攝影影像中診斷乳癌。它利用深度學習模型組合 (Ensemble) 來預測惡性程度，並透過視覺化熱圖 (Grad-CAM) 和使用 RAG（檢索增強生成）生成的臨床報告提供具解釋性的結果。

---

## 2. 技術棧 (Technology Stack)

### **前端 (使用者介面)**
*   **框架**: [React](https://react.dev/) (v18)
*   **構建工具**: [Vite](https://vitejs.dev/)
*   **樣式**:
    *   **Tailwind CSS**: 用於響應式佈局和實用優先 (Utility-first) 的樣式設計。
    *   **Framer Motion**: 用於流暢、高效能的 UI 動畫（分頁切換、頁面轉場）。
    *   **毛玻璃效果 (Glassmorphism)**: 自定義 CSS 以實現現代化、半透明的「玻璃」審美感。
*   **狀態管理**: React `useState` / `useEffect`。
*   **HTTP 客戶端**: `Axios` 用於與 Python 後端進行通訊。

### **後端 (AI 引擎)**
*   **框架**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
*   **深度學習**:
    *   **TensorFlow / Keras**: 用於加載和運行神經網路。
    *   **OpenCV (`cv2`)**: 用於影像預處理（CLAHE、調整大小）。
    *   **PyTorch**: 用於特定的 RAG 向量嵌入（透過 `sentence-transformers`）。
*   **PDF 生成**: `fpdf2` 用於動態創建專業且結構化的 PDF 報告。
*   **RAG 系統**: 自定義實作，使用從 `knowledge_base.md` 進行的本地知識檢索。

---

## 3. 託管與基礎設施

### **前端部署**
*   **平台**: **Vercel**
*   **URL**: `https://medvisionai-project.vercel.app` (示例)
*   **使用功能**: 邊緣快取 (Edge caching)、來自 Git 的持續部署 (CD)。

### **後端部署**
*   **平台**: **Hugging Face Spaces**
*   **SDK**: Docker
*   **硬體**: 16GB RAM CPU Basic (同時加載 ResNet+VGG+CNN 所需)。
*   **容器**: 基於 `python:3.10-slim` 的自定義 `Dockerfile`。
*   **原因**: Vercel/Netlify 的伺服器端函式 (Functions) 有 50MB 的大小限制且記憶體較低，不足以運行 TensorFlow 模型。Hugging Face Spaces 允許運行較大的容器化應用程式。

---

## 4. AI 與模型架構

### **「組合模型 (Ensemble)」方法**
系統不依賴單一模型，而是使用三種不同的架構來提供「第二意見 (Second Opinion)」生態系統：

1.  **CNN + CLAHE (專家系統)**:
    *   **架構**: 自定義卷積神經網路 (CNN)。
    *   **輸入**: 使用 **CLAHE** (對比度受限自適應直方圖均衡化) 預處理過的影像。
    *   **優勢**: 對結構異常極其敏感；最擅長檢測緻密組織中的腫塊。

2.  **ResNet50 (通用專家)**:
    *   **架構**: 深度殘差學習 (Deep Residual Learning)。
    *   **優勢**: 在標準數據集上具有高準確度；對影像縮放/角度的變化具有魯棒性。

3.  **VGG16 / VGG19 (細節發現者)**:
    *   **架構**: 具有小型感受野的極深網路。
    *   **優勢**: 擅長提取細微的紋理，例如微小鈣化點 (Micro-calcifications)。

### **可解釋人工智慧 (XAI)**
*   **Grad-CAM** (梯度加權類別激活映射):
    *   系統從最後一個卷積層提取梯度以生成「熱圖」。
    *   **紅色區域**: 表示模型發揮作用的區域（例如：腫瘤邊界）。
    *   **藍色區域**: 背景組織。
    *   *目的*: 這證明了模型並非「作弊」（例如：依靠浮水印判斷），而是真正專注於病灶。

---

## 5. RAG (檢索增強生成)
系統並非使用聊天機器人，而是採用了**安全、確定性的 RAG 流程**：

1.  **診斷**: 模型預測「癌症 (98%)」。
2.  **檢metadata**: 系統在 `knowledge_base.md` 中搜索與「惡性乳房攝影特徵」相關的情境。
3.  **生成**: 它編譯一份結合以下內容的臨床報告：
    *   特定模型的信心分數。
    *   Grad-CAM 的發現（例如：「焦點高強度激活」）。
    *   檢索到的臨床指南（例如：「建議 BI-RADS 4/5 評估」）。
4.  **結果**: 具備引用來源、醫學根據的文本，無幻覺 (Hallucination) 問題。

---

## 6. 目錄結構
```
medvision-ai/
├── api/                   # 後端程式碼
│   ├── main.py            # FastAPI 入口點與端點
│   ├── rag_engine.py      # 知識庫檢索邏輯
│   └── knowledge_base.md  # 醫學百科檔案
├── frontend/              # 前端程式碼
│   └── src/
│       └── App.jsx        # 主要 React 邏輯
├── requirements.txt       # Python 依賴項
└── Dockerfile             # Hugging Face 的伺服器配置
```
