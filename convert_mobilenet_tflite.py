"""
Convert MobileNet Feature Extractor to TFLite
Creates both Float32 and INT8 quantized versions
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image

print("=" * 70)
print("TFLITE CONVERSION - MOBILENET FEATURE EXTRACTOR")
print("=" * 70)

# Configuration
MODEL_PATH = "mobilenet_feature_extractor.h5"
OUTPUT_DIR = "tflite_models"
DATASET_DIR = "cfv_viewpoint_train"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Load Model
# ============================================================================

print("\nLoading feature extractor...")
model = keras.models.load_model(MODEL_PATH)
print(f"[OK] Model loaded: {MODEL_PATH}")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# ============================================================================
# 2. Convert to Float32 TFLite
# ============================================================================

print("\n" + "=" * 70)
print("Converting to Float32 TFLite...")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # No optimization for float32

tflite_float32 = converter.convert()

# Save
float32_path = os.path.join(OUTPUT_DIR, "mobilenet_feature_extractor_float32.tflite")
with open(float32_path, 'wb') as f:
    f.write(tflite_float32)

float32_size = os.path.getsize(float32_path) / (1024 * 1024)
print(f"\n[OK] Float32 TFLite saved!")
print(f"  Path: {float32_path}")
print(f"  Size: {float32_size:.2f} MB")

# ============================================================================
# 3. Prepare Representative Dataset for INT8
# ============================================================================

print("\n" + "=" * 70)
print("Preparing representative dataset for INT8 quantization...")
print("=" * 70)

def representative_dataset_gen():
    """Generate representative samples for INT8 quantization"""
    dataset_path = Path(DATASET_DIR)
    
    # Collect sample images
    image_paths = []
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))[:15]  # 15 per class
            image_paths.extend(images)
    
    image_paths = image_paths[:100]  # Limit to 100
    print(f"Using {len(image_paths)} representative images")
    
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img).astype(np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            yield [img_array]
        except:
            continue

# ============================================================================
# 4. Convert to INT8 TFLite
# ============================================================================

print("\n" + "=" * 70)
print("Converting to INT8 TFLite...")
print("=" * 70)

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen

# Full integer quantization
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.int8

print("\nQuantizing... (this may take a minute)")
tflite_int8 = converter_int8.convert()

# Save
int8_path = os.path.join(OUTPUT_DIR, "mobilenet_feature_extractor_int8.tflite")
with open(int8_path, 'wb') as f:
    f.write(tflite_int8)

int8_size = os.path.getsize(int8_path) / (1024 * 1024)
print(f"\n[OK] INT8 TFLite saved!")
print(f"  Path: {int8_path}")
print(f"  Size: {int8_size:.2f} MB")

# ============================================================================
# 5. Verify Models
# ============================================================================

print("\n" + "=" * 70)
print("Verifying TFLite Models")
print("=" * 70)

# Test Float32
print("\n[Float32 Model]")
interpreter_f32 = tf.lite.Interpreter(model_path=float32_path)
interpreter_f32.allocate_tensors()

input_details_f32 = interpreter_f32.get_input_details()
output_details_f32 = interpreter_f32.get_output_details()

print(f"  Input: {input_details_f32[0]['shape']} | {input_details_f32[0]['dtype']}")
print(f"  Output: {output_details_f32[0]['shape']} | {output_details_f32[0]['dtype']}")

# Test INT8
print("\n[INT8 Model]")
interpreter_int8 = tf.lite.Interpreter(model_path=int8_path)
interpreter_int8.allocate_tensors()

input_details_int8 = interpreter_int8.get_input_details()
output_details_int8 = interpreter_int8.get_output_details()

print(f"  Input: {input_details_int8[0]['shape']} | {input_details_int8[0]['dtype']}")
print(f"  Output: {output_details_int8[0]['shape']} | {output_details_int8[0]['dtype']}")

# Test inference
print("\n[Testing Inference]")
test_input_f32 = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
test_input_f32 = (test_input_f32 - 0.5) * 2  # MobileNet preprocessing

interpreter_f32.set_tensor(input_details_f32[0]['index'], test_input_f32)
interpreter_f32.invoke()
output_f32 = interpreter_f32.get_tensor(output_details_f32[0]['index'])

print(f"  Float32 output shape: {output_f32.shape}")
print(f"  Float32 output range: [{output_f32.min():.4f}, {output_f32.max():.4f}]")

test_input_int8 = np.random.randint(0, 256, (1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
interpreter_int8.set_tensor(input_details_int8[0]['index'], test_input_int8)
interpreter_int8.invoke()
output_int8 = interpreter_int8.get_tensor(output_details_int8[0]['index'])

print(f"  INT8 output shape: {output_int8.shape}")
print(f"  INT8 output range: [{output_int8.min()}, {output_int8.max()}]")

print("\n[OK] Both models verified!")

# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "=" * 70)
print("TFLITE CONVERSION COMPLETE")
print("=" * 70)

print(f"\nModel Comparison:")
print(f"  Float32: {float32_size:.2f} MB")
print(f"  INT8:    {int8_size:.2f} MB")
print(f"  Reduction: {((float32_size - int8_size) / float32_size * 100):.1f}%")

print(f"\nGenerated Files (in {OUTPUT_DIR}/):")
print("  - mobilenet_feature_extractor_float32.tflite")
print("  - mobilenet_feature_extractor_int8.tflite")

print("\n[OK] Ready for mobile/edge deployment!")
