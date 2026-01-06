import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)

def get_base_cells(title, model_type):
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# Breast Cancer Classification: {title}\n", f"This notebook implements the {model_type} model for breast cancer classification."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import cv2\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import tensorflow as tf\n",
                "import sys\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import classification_report, confusion_matrix\n",
                "from tensorflow.keras.models import Sequential, Model\n",
                "from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D\n",
                "from tensorflow.keras.utils import to_categorical\n",
                "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
                "from tensorflow.keras.regularizers import l2\n",
                "from tensorflow.keras.optimizers import Adam, SGD\n",
                "\n",
                "sys.path.append('..')\n",
                "from report_generator import ReportGenerator"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "DATASET_ROOT = '../dataset'\n",
                f"OUTPUT_DIR = 'output_{title.lower().replace(' ', '_')}'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "\n",
                "def load_data(data_dir):\n",
                "    images, labels = [], []\n",
                "    for ds in ['Augmented Dataset', 'Original Dataset']:\n",
                "        path = os.path.join(data_dir, ds)\n",
                "        if not os.path.exists(path): path = data_dir\n",
                "        for label in ['Cancer', 'Non-Cancer']:\n",
                "            l_path = os.path.join(path, label)\n",
                "            if os.path.isdir(l_path):\n",
                "                for f in os.listdir(l_path):\n",
                "                    if f.lower().endswith(('.png','.jpg','.jpeg')):\n",
                "                        img = cv2.imread(os.path.join(l_path, f), cv2.IMREAD_GRAYSCALE)\n",
                "                        if img is not None:\n",
                "                            img = cv2.resize(img, (128,128))\n",
                "                            img = cv2.equalizeHist(img)\n",
                "                            images.append(img/255.0)\n",
                "                            labels.append(1 if label=='Cancer' else 0)\n",
                "    return np.array(images), np.array(labels)\n",
                "\n",
                "def apply_clahe(images):\n",
                "    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\n",
                "    processed = [clahe.apply((img*255).astype(np.uint8))/255.0 for img in images]\n",
                "    return np.array(processed)\n",
                "\n",
                "images, labels = load_data(DATASET_ROOT)\n",
                "X = apply_clahe(images).reshape(-1,128,128,1)\n",
                "y = to_categorical(labels, 2)\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
            ]
        }
    ]

# Helper to add code cell with all required keys
def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

# --- CNN+CLAHE ---
cnn_cells = get_base_cells("Custom CNN + CLAHE", "Custom CNN")
cnn_cells += [
    code_cell("def build_model():\n    model = Sequential([\n        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1), name='conv_idx_0'),\n        BatchNormalization(), MaxPooling2D((2, 2)),\n        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_idx_1'),\n        BatchNormalization(), MaxPooling2D((2, 2)),\n        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_idx_2'),\n        BatchNormalization(), MaxPooling2D((2, 2)),\n        Flatten(), Dense(256, activation='relu'), Dropout(0.5),\n        Dense(128, activation='relu'), Dropout(0.5),\n        Dense(2, activation='softmax')\n    ])\n    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n    return model\n\nmodel = build_model()\nhistory = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))")
]

# --- ResNet50 ---
resnet_cells = get_base_cells("ResNet50", "Transfer Learning")
resnet_cells += [
    code_cell("from tensorflow.keras.applications import ResNet50\ndef build_model():\n    base = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')\n    for layer in base.layers[:-4]: layer.trainable = False\n    model = Sequential([\n        Input(shape=(128, 128, 1)),\n        Conv2D(3, (3, 3), padding='same', activation='relu', name='channel_conversion'),\n        base, GlobalAveragePooling2D(),\n        Dense(512, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.4),\n        Dense(2, activation='softmax')\n    ])\n    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])\n    return model, base\n\nmodel, base_model = build_model()\nhistory = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))")
]

# --- VGG16 ---
vgg16_cells = get_base_cells("VGG16", "Transfer Learning")
vgg16_cells += [
    code_cell("from tensorflow.keras.applications import VGG16\ndef build_model():\n    base = VGG16(input_shape=(128, 128, 3), include_top=False, weights='imagenet')\n    for layer in base.layers[:-4]: layer.trainable = False\n    model = Sequential([\n        Input(shape=(128, 128, 1)),\n        Conv2D(3, (3, 3), padding='same', activation='relu'),\n        base, GlobalAveragePooling2D(),\n        Dense(512, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.4),\n        Dense(2, activation='softmax')\n    ])\n    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])\n    return model\n\nmodel = build_model()\nhistory = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))")
]

# --- VGG19 ---
vgg19_cells = get_base_cells("VGG19", "Transfer Learning")
vgg19_cells += [
    code_cell("from tensorflow.keras.applications import VGG19\ndef build_model():\n    base = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')\n    for layer in base.layers[:-4]: layer.trainable = False\n    model = Sequential([\n        Input(shape=(128, 128, 1)),\n        Conv2D(3, (3, 3), padding='same', activation='relu'),\n        base, GlobalAveragePooling2D(),\n        Dense(512, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.4),\n        Dense(2, activation='softmax')\n    ])\n    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])\n    return model\n\nmodel = build_model()\nhistory = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))")
]

os.makedirs('notebooks', exist_ok=True)
create_notebook('notebooks/cnn_clahe.ipynb', cnn_cells)
create_notebook('notebooks/resnet50.ipynb', resnet_cells)
create_notebook('notebooks/vgg16.ipynb', vgg16_cells)
create_notebook('notebooks/vgg19.ipynb', vgg19_cells)
print(f"Created all notebooks in {os.path.abspath('notebooks')} folder.")
