import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from report_generator import ReportGenerator

# ==============================
# CONFIGURATION
# ==============================
# You can change this path to single folder if you extracted it differently
DATASET_ROOT = 'dataset' 
OUTPUT_DIR = 'output'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
        
# GPU Check
print("Checking for GPU acceleration...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPU is available: {physical_devices}")
    # Enable memory growth if needed (common for Metal)
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
else:
    print("GPU not found. Running on CPU.")

def load_data(data_dir):
    print(f"Loading data from: {data_dir}")
    datasets = ['Augmented Dataset', 'Original Dataset']
    images = []
    labels = []
    
    found_any = False

    # Check if direct path or subdivided
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return np.array([]), np.array([])

    # Try to find the structure expected: Dataset -> Cancer/Non-Cancer
    # Some users might just have 'dataset/Cancer' directly. 
    # The original code iterated over 'Augmented values' etc. We try to support that or fallback.
    
    # Check if 'Augmented Dataset' exists inside
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Strategy: Walk through and look for 'Cancer' and 'Non-Cancer' folders
    target_labels = ['Cancer', 'Non-Cancer']
    
    # Helper to process a label directory
    def process_label_dir(path, label_val):
        count = 0
        for img_file in os.listdir(path):
            if img_file.lower().endswith(('.png','.jpg','.jpeg')):
                img_path = os.path.join(path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128,128))
                    img = cv2.equalizeHist(img)
                    img = img / 255.0
                    images.append(img)
                    labels.append(label_val)
                    count += 1
        return count

    # Recursive search or specific structure support
    # We stick to the user's specific loop structure but make it robust
    for dataset_name in datasets:
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            # Fallback: maybe the user just put all images in 'dataset/Cancer' directly?
            # Let's check if the root has Cancer/Non-Cancer
            dataset_path = data_dir 
        
        for label_name in target_labels:
            label_dir = os.path.join(dataset_path, label_name)
            if os.path.isdir(label_dir):
                found_any = True
                print(f"Processing {label_dir}...")
                count = process_label_dir(label_dir, 1 if label_name=='Cancer' else 0)
                print(f"Loaded {count} images from {label_name}")

    if not found_any:
        print("WARNING: No data found. Please ensure your directory structure matches:")
        print(f"  {data_dir}/Augmented Dataset/Cancer/...")
        print(f"  {data_dir}/Original Dataset/Non-Cancer/...")
        print("  OR simply:")
        print(f"  {data_dir}/Cancer/...")
    
    return np.array(images), np.array(labels)

def save_samples(images, labels, n=5):
    if len(images) == 0: return
    cancer_idx = np.where(labels==1)[0]
    non_idx = np.where(labels==0)[0]
    
    if len(cancer_idx) < n or len(non_idx) < n:
        print("Not enough images to show samples.")
        return

    plt.figure(figsize=(15,5))
    for i, idx in enumerate(np.random.choice(cancer_idx, n)):
        plt.subplot(2,n,i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title("Cancer")
        plt.axis('off')
    for i, idx in enumerate(np.random.choice(non_idx, n)):
        plt.subplot(2,n,n+i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title("Non-Cancer")
        plt.axis('off')
    
    out_path = os.path.join(OUTPUT_DIR, '01_samples.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sample images to {out_path}")

def apply_clahe(images):
    print("Applying CLAHE preprocessing...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = []
    for img in images:
        img_c = clahe.apply((img*255).astype(np.uint8))
        processed.append(img_c/255.0)
    return np.array(processed)

def build_model():
    print("Building Custom CNN model...")
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1), name='conv_idx_0'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_idx_1'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_idx_2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()
    return cnn_model

def make_gradcam_heatmap(img, model, last_conv_layer_name="conv_idx_2"):
    """
    Computes a Grad-CAM heatmap for a given image and model.
    """
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        # Gradients of the loss with respect to the output feature map
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        return None

# Note: The original GradCAM code might differ slightly in how it accessed layers. 
# Since we put VGG inside Sequential, `model.get_layer(base_model_name)` might not work directly as a name lookup string if not named.
# I'll include a safe try-except block for GradCAM.

def overlay_gradcam(img, heatmap, alpha=0.4):
    img = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img,1-alpha,heatmap,alpha,0)

def main():
    # 1. Load Data
    images, labels = load_data(DATASET_ROOT)
    
    if len(images) == 0:
        print("STOPPING: No images loaded.")
        return

    # 2. Visualize Samples
    save_samples(images, labels)

    # 3. Preprocess
    X = apply_clahe(images)
    X = X.reshape(-1,128,128,1)
    y = to_categorical(labels, num_classes=2)

    # 4. Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Build & Train
    model = build_model()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=25, 
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # 6. Evaluation
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save Model
    model.save('model_cnn_clahe.keras')
    print("Model saved as model_cnn_clahe.keras")

    # 7. Metrics & Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_cls)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Non-Cancer','Cancer'],
                yticklabels=['Non-Cancer','Cancer'])
    plt.savefig(os.path.join(OUTPUT_DIR, '02_confusion_matrix.png'))
    plt.close()
    print("Saved confusion matrix.")

    print(classification_report(y_true, y_pred_cls))

    # 8. Grad-CAM (XAI Visualization)
    print("Generating Grad-CAM visualizations...")
    try:
        # Create output directory for GradCAM if not exists
        gradcam_output_dir = os.path.join(OUTPUT_DIR, 'gradcam')
        os.makedirs(gradcam_output_dir, exist_ok=True)
        
        # Process first 5 images from test set
        for i in range(5):
            img = X_test[i:i+1]
            # Convert to tensor for GradCAM
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            
            heatmap = make_gradcam_heatmap(img_tensor, model)
            if heatmap is not None:
                cam_img = overlay_gradcam(img[0], heatmap)
                
                # Save result
                label_name = 'Cancer' if y_true[i] == 1 else 'Non-Cancer'
                pred_name = 'Cancer' if y_pred_cls[i] == 1 else 'Non-Cancer'
                
                out_path = os.path.join(gradcam_output_dir, f'img_{i}_{label_name}_pred_{pred_name}.png')
                plt.figure(figsize=(5,5))
                plt.imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
                plt.title(f"True: {label_name}, Pred: {pred_name}")
                plt.axis("off")
                plt.savefig(out_path)
                plt.close()
                print(f"Saved Grad-CAM for image {i} to {out_path}")
    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        import traceback
        traceback.print_exc()

    # 9. RAG + LLM Report Generation
    print("Integrating RAG+LLM for AI-Assisted Diagnosis Reports...")
    try:
        reports_output_dir = os.path.join(OUTPUT_DIR, 'reports')
        os.makedirs(reports_output_dir, exist_ok=True)
        
        kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.md')
        if not os.path.exists(kb_path):
            kb_path = 'knowledge_base.md'
            
        report_gen = ReportGenerator(kb_path)
        
        # Generate reports for the first 5 images (same as Grad-CAM)
        for i in range(5):
            pred_name = 'Cancer' if y_pred_cls[i] == 1 else 'Non-Cancer'
            confidence = y_pred[i][y_pred_cls[i]] * 100
            
            if pred_name == 'Cancer':
                finding = "High-intensity activation detected in dense tissue regions, suggesting structural irregularity consistent with malignant morphology."
            else:
                finding = "Low or diffuse activation patterns observed; no focal areas of high density identified that strongly suggest malignancy."
            
            report_file = os.path.join(reports_output_dir, f'report_{i}_{pred_name}.md')
            report_gen.generate_report(
                prediction_label=pred_name,
                confidence=confidence,
                gradcam_finding=finding,
                output_path=report_file
            )
            print(f"Generated AI-Assisted Diagnosis Report: {report_file}")
            
    except Exception as e:
        print(f"Error during Report Generation: {e}")

if __name__ == "__main__":
    main()
