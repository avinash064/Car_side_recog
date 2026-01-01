"""
Convert trained Keras model to TFLite format.
Supports both float32 and INT8 quantization.
"""

import os
import tensorflow as tf
from pathlib import Path
import numpy as np

# Configuration
SAVEDMODEL_PATH = "viewpoint_model_savedmodel"
TFLITE_DIR = "tflite_models"

# For INT8 quantization (optional)
REPRESENTATIVE_DATA_DIR = r"C:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog\cfv_viewpoint_train"
IMG_SIZE = 224


def representative_dataset_gen():
    """
    Generate representative dataset for INT8 quantization.
    Samples images from training data.
    """
    from tensorflow import keras
    import glob
    
    # Collect sample images from all classes
    image_paths = []
    for class_dir in Path(REPRESENTATIVE_DATA_DIR).iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            image_paths.extend(images[:20])  # 20 images per class
    
    # Limit to 100 total images
    image_paths = image_paths[:100]
    
    print(f"Using {len(image_paths)} representative images for quantization")
    
    for img_path in image_paths:
        img = keras.preprocessing.image.load_img(
            img_path, 
            target_size=(IMG_SIZE, IMG_SIZE)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        yield [img_array.astype(np.float32)]


def convert_to_float32():
    """Convert to float32 TFLite model."""
    print("\n" + "=" * 60)
    print("Converting to Float32 TFLite")
    print("=" * 60)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_PATH)
    tflite_model = converter.convert()
    
    output_path = os.path.join(TFLITE_DIR, 'viewpoint_classifier_float32.tflite')
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Float32 model saved: {output_path}")
    print(f"  Size: {size_mb:.2f} MB")
    
    return output_path


def convert_to_int8():
    """Convert to INT8 quantized TFLite model."""
    print("\n" + "=" * 60)
    print("Converting to INT8 TFLite (quantized)")
    print("=" * 60)
    
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_PATH)
        
        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        output_path = os.path.join(TFLITE_DIR, 'viewpoint_classifier_int8.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] INT8 model saved: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"⚠ INT8 conversion failed: {e}")
        print("  Skipping INT8 quantization.")
        return None


def verify_model(tflite_path):
    """Verify TFLite model can be loaded and check input/output specs."""
    print("\n" + "-" * 60)
    print(f"Verifying: {os.path.basename(tflite_path)}")
    print("-" * 60)
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input specs:")
    for detail in input_details:
        print(f"  Shape: {detail['shape']}, Type: {detail['dtype']}")
    
    print("Output specs:")
    for detail in output_details:
        print(f"  Shape: {detail['shape']}, Type: {detail['dtype']}")
    
    print("[OK] Model verified successfully")


def main():
    print("=" * 60)
    print("TFLite Model Conversion")
    print("=" * 60)
    
    # Check if SavedModel exists
    if not os.path.exists(SAVEDMODEL_PATH):
        print(f"[ERROR] SavedModel not found at: {SAVEDMODEL_PATH}")
        print("Please run training script first.")
        return
    
    # Create output directory
    os.makedirs(TFLITE_DIR, exist_ok=True)
    
    # Convert to Float32
    float32_path = convert_to_float32()
    verify_model(float32_path)
    
    # Convert to INT8 (optional)
    if os.path.exists(REPRESENTATIVE_DATA_DIR):
        int8_path = convert_to_int8()
        if int8_path:
            verify_model(int8_path)
    else:
        print(f"\n[WARNING] Representative data not found at: {REPRESENTATIVE_DATA_DIR}")
        print("  Skipping INT8 quantization.")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Models saved to: {TFLITE_DIR}")


if __name__ == '__main__':
    main()
