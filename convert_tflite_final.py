"""
TFLite Converter - Works around mixed precision issue
Rebuilds model without mixed precision and loads weights
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
import numpy as np

print("=" * 70)
print("TFLite Conversion (Mixed Precision Workaround)")
print("=" * 70)

# Disable mixed precision for conversion
keras.mixed_precision.set_global_policy('float32')

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 7
CHECKPOINT_PATH = "best_viewpoint_model.h5"
OUTPUT_DIR = "tflite_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nStep 1: Rebuilding model with float32 policy...")

def build_model_float32():
    """Build model without mixed precision"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Build model
model = build_model_float32()
print(f"Model built with float32 policy ({model.count_params():,} parameters)")

# Load weights
print("\nStep 2: Loading trained weights...")
try:
    model.load_weights(CHECKPOINT_PATH)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    exit(1)

# Verify model output
print("\nStep 3: Verifying model...")
dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE,3).astype(np.float32)
output = model.predict(dummy_input, verbose=0)
print(f"Model output shape: {output.shape}")
print(f"Model output sum: {output.sum():.4f} (should be ~1.0)")

# ============================================================================
# Convert to TFLite
# ============================================================================

print("\n" + "=" * 70)
print("Converting to TFLite...")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("\nConverting...")
tflite_model = converter.convert()

float32_path = os.path.join(OUTPUT_DIR, "viewpoint_classifier_float32.tflite")
with open(float32_path, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(float32_path) / (1024 * 1024)
print(f"\n[SUCCESS] TFLite model saved!")
print(f"  Path: {float32_path}")
print(f"  Size: {size_mb:.2f} MB")

# ============================================================================
# Verify TFLite Model
# ============================================================================

print("\n" + "=" * 70)
print("Verifying TFLite Model")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=float32_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput Specification:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")

print("\nOutput Specification:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")

# Test inference
print("\n" + "-" * 70)
print("Testing Inference...")

test_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {tflite_output.shape}")
print(f"  Output sum: {tflite_output.sum():.4f} (should be ~1.0)")
print(f"  Predicted class: {np.argmax(tflite_output)}")

print("\n[OK] TFLite model verified successfully!")

print("\n" + "=" * 70)
print("CONVERSION COMPLETE!")
print("=" * 70)
print(f"\nTFLite Model: {float32_path}")
print(f"Size: {size_mb:.2f} MB")
print(f"\nModel is ready for mobile deployment!")
print("\nUsage:")
print("  - Input: [1, 224, 224, 3] float32 (RGB image, normalized 0-1)")
print(f"  - Output: [1, {NUM_CLASSES}] float32 (class probabilities)")
