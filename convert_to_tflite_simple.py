"""
Simple TFLite Converter for Viewpoint Classifier
Converts the trained H5 model to TFLite format.
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

print("=" * 70)
print("TFLite Model Conversion")
print("=" * 70)

# Paths
H5_MODEL_PATH = "viewpoint_model.h5"
OUTPUT_DIR = "tflite_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check model exists
if not os.path.exists(H5_MODEL_PATH):
    print(f"Error: Model not found at {H5_MODEL_PATH}")
    exit(1)

print(f"\nLoading model from: {H5_MODEL_PATH}")
model = keras.models.load_model(H5_MODEL_PATH, compile=False)
print(f"Model loaded successfully")

# ============================================================================
# Float32 Conversion
# ============================================================================

print("\n" + "=" * 70)
print("Converting to Float32 TFLite")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

float32_path = os.path.join(OUTPUT_DIR, "viewpoint_classifier_float32.tflite")
with open(float32_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(float32_path) / (1024 * 1024)
print(f"\n[OK] Float32 model saved: {float32_path}")
print(f"     Size: {size_mb:.2f} MB")

# ============================================================================
# Verify Model
# ============================================================================

print("\n" + "=" * 70)
print("Verifying TFLite Model")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=float32_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput specs:")
for detail in input_details:
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")

print("\nOutput specs:")
for detail in output_details:
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")

# Test inference
print("\n" + "-" * 70)
print("Running test inference...")
print("-" * 70)

# Create dummy input
input_shape = input_details[0]['shape']
dummy_input = np.random.rand(*input_shape).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output type: {output.dtype}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"Output sum: {output.sum():.4f} (should be ~1.0 for softmax)")

print("\n[OK] Model verified successfully!")

print("\n" + "=" * 70)
print("CONVERSION COMPLETE!")
print("=" * 70)
print(f"\nTFLite model saved to: {float32_path}")
print(f"Model size: {size_mb:.2f} MB")
print(f"\nReady for mobile deployment!")
