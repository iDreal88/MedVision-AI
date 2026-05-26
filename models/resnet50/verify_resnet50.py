import os
import numpy as np
import tensorflow as tf
from main import build_model, make_gradcam_heatmap, overlay_gradcam

def test_architecture():
    print("Testing Model Architecture...")
    try:
        model, base_model = build_model()
        print("Model built successfully.")
        
        # Check input shape
        if model.input_shape != (None, 128, 128, 1):
            print(f"Error: Incorrect input shape {model.input_shape}")
            return False
            
        print("Input shape verified.")
        return True
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradcam():
    print("\nTesting Grad-CAM Logic...")
    try:
        model, base_model = build_model()
        
        # Create dummy image
        dummy_img = np.random.random((1, 128, 128, 1)).astype(np.float32)
        img_tensor = tf.convert_to_tensor(dummy_img)
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_tensor, model, base_model)
        
        if heatmap is not None:
            print("Grad-CAM heatmap generated successfully.")
            print(f"Heatmap shape: {heatmap.shape}")
            
            # Test overlay
            cam_img = overlay_gradcam(dummy_img[0], heatmap)
            print("Grad-CAM overlay generated successfully.")
            return True
        else:
            print("Failed to generate Grad-CAM heatmap.")
            return False
    except Exception as e:
        print(f"Error in Grad-CAM test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    arch_ok = test_architecture()
    gradcam_ok = test_gradcam()
    
    if arch_ok and gradcam_ok:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        exit(1)
