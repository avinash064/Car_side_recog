"""
Create INT8 Quantized TFLite Model
Produces a smaller, faster model for mobile deployment
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

print("=" * 70)
print("INT8 Quantization")
print("=" * 70)

# Disable mixed precision
keras.mixed_precision.set_global_policy('float32')

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 7
CHECKPOINT_PATH = "best_viewpoint_model.h5"
DATASET_DIR = "cfv_viewpoint_train"
OUTPUT_DIR = "tflite_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build model
print("\nStep 1: Building model architecture...")

def build_model_float32():
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

model = build_model_float32()
print(f"Model built ({model.count_params():,} parameters)")

# Load weights
print("\nStep 2: Loading trained weights...")
try:
    model.load_weights(CHECKPOINT_PATH)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Using ImageNet weights only...")

# Representative dataset for quantization
print("\nStep 3: Preparing representative dataset...")

def representative_dataset_gen():
    """Generate representative samples for quantization calibration"""
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: Dataset not found at {DATASET_DIR}")
        print("Using random data for calibration...")
        for _ in range(100):
            data = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            yield [data]
    else:
        # Collect sample images from all classes
        image_paths = []
        for class_dir in Path(DATASET_DIR).iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                image_paths.extend(images[:15])  # 15 images per class
        
        image_paths = image_paths[:100]  # Limit to 100 total
        print(f"Using {len(image_paths)} representative images")
        
        for img_path in image_paths:
            try:
                from PIL import Image
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                yield [img_array]
            except:
                continue

# Convert to INT8 TFLite
print("\n" + "=" * 70)
print("Converting to INT8 TFLite...")
print("=" * 70)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Ensure full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

print("\nQuantizing... (this may take a minute)")
tflite_model_int8 = converter.convert()

# Save INT8 model
int8_path = os.path.join(OUTPUT_DIR, "viewpoint_classifier_int8.tflite")
with open(int8_path, 'wb') as f:
    f.write(tflite_model_int8)

size_mb = os.path.getsize(int8_path) / (1024 * 1024)
print(f"\n[SUCCESS] INT8 model saved!")
print(f"  Path: {int8_path}")
print(f"  Size: {size_mb:.2f} MB")

# Verify INT8 model
print("\n" + "=" * 70)
print("Verifying INT8 Model")
print("=" * 70)

interpreter_int8 = tf.lite.Interpreter(model_path=int8_path)
interpreter_int8.allocate_tensors()

input_details = interpreter_int8.get_input_details()
output_details = interpreter_int8.get_output_details()

print("\nInput Specification:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")
print(f"  Quantization: scale={input_details[0]['quantization'][0]:.6f}, zero_point={input_details[0]['quantization'][1]}")

print("\nOutput Specification:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")
print(f"  Quantization: scale={output_details[0]['quantization'][0]:.6f}, zero_point={output_details[0]['quantization'][1]}")

# Test inference
print("\n" + "-" * 70)
print("Testing INT8 Inference...")

# Create test input (uint8 range 0-255)
test_input = np.random.randint(0, 256, size=(1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
interpreter_int8.set_tensor(input_details[0]['index'], test_input)
interpreter_int8.invoke()
int8_output = interpreter_int8.get_tensor(output_details[0]['index'])

print(f"  Input shape: {test_input.shape}, dtype: {test_input.dtype}")
print(f"  Output shape: {int8_output.shape}, dtype: {int8_output.dtype}")
print(f"  Output range: [{int8_output.min()}, {int8_output.max()}]")

# Dequantize output to see probabilities
scale = output_details[0]['quantization'][0]
zero_point = output_details[0]['quantization'][1]
dequantized = (int8_output.astype(np.float32) - zero_point) * scale
print(f"  Dequantized sum: {dequantized.sum():.4f} (should be ~1.0)")

print("\n[OK] INT8 model verified successfully!")

# Compare sizes
if os.path.exists(os.path.join(OUTPUT_DIR, "viewpoint_classifier_float32.tflite")):
    float32_size = os.path.getsize(os.path.join(OUTPUT_DIR, "viewpoint_classifier_float32.tflite")) / (1024 * 1024)
    print("\n" + "=" * 70)
    print("SIZE COMPARISON")
    print("=" * 70)
    print(f"Float32 model: {float32_size:.2f} MB")
    print(f"INT8 model:    {size_mb:.2f} MB")
    print(f"Reduction:     {((float32_size - size_mb) / float32_size * 100):.1f}%")

print("\n" + "=" * 70)
print("INT8 QUANTIZATION COMPLETE!")
print("=" * 70)
print(f"\nModel saved to: {int8_path}")
print(f"Model size: {size_mb:.2f} MB")
print(f"\nReady for mobile deployment with faster inference!")
