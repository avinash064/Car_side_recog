"""
Resume Training Script for 7-Class Canonical Vehicle Viewpoint Classifier
=========================================================================
Resumes from best_viewpoint_model.h5 checkpoint with reduced batch size to avoid OOM.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ============================================================================
# Configuration - REDUCED BATCH SIZE to avoid OOM
# ============================================================================

DATASET_DIR = "cfv_viewpoint_train"
IMG_SIZE = 224
BATCH_SIZE = 32  # REDUCED from 64 to avoid GPU OOM
NUM_CLASSES = 7

# Resume from checkpoint - only fine-tuning epochs remain
FINETUNE_EPOCHS = 45
FINETUNE_LR = 2e-4

# Checkpoint and output paths
CHECKPOINT_PATH = "best_viewpoint_model.h5"
SAVEDMODEL_DIR = "viewpoint_model_savedmodel"
H5_MODEL_PATH = "viewpoint_model.h5"
CLASS_MAP_PATH = "class_map.json"

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

print("=" * 80)
print("RESUMING VIEWPOINT CLASSIFIER TRAINING FROM CHECKPOINT")
print("=" * 80)
print(f"\nBatch size reduced to {BATCH_SIZE} to prevent OOM")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# ============================================================================
# Data Loading
# ============================================================================

print("\n[1/4] Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.85, 1.15],
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

class_indices = train_generator.class_indices
class_map = {v: k for k, v in class_indices.items()}

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ============================================================================
# Compute Class Weights
# ============================================================================

print("\n[2/4] Computing class weights...")

train_labels = []
for class_name, class_idx in class_indices.items():
    class_dir = os.path.join(DATASET_DIR, class_name)
    num_samples = len(os.listdir(class_dir))
    train_samples = int(num_samples * 0.8)
    train_labels.extend([class_idx] * train_samples)

train_labels = np.array(train_labels)
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print("Class weights computed")

# Build model architecture (same as original)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers

def build_model(trainable_backbone=True):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = trainable_backbone
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Mixed precision: ensure float32 output
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    return keras.Model(inputs, outputs)

# Build model
model = build_model(trainable_backbone=True)
print(f"Model architecture built ({model.count_params():,} parameters)")

# Load weights from checkpoint
try:
    model.load_weights(CHECKPOINT_PATH)
    print(f"Weights loaded successfully from {CHECKPOINT_PATH}")
except Exception as e:
    print(f"Warning: Could not load weights: {e}")
    print("Continuing with pre-trained ImageNet weights only...")

# ============================================================================
# Fine-tuning Stage
# ============================================================================

print(f"\n[4/4] Resuming Fine-tuning ({FINETUNE_EPOCHS} epochs)...")

optimizer_finetune = keras.optimizers.Adam(
    learning_rate=FINETUNE_LR
)

model.compile(
    optimizer=optimizer_finetune,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

finetune_callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_viewpoint_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINETUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=finetune_callbacks,
    verbose=1
)

print("Fine-tuning completed")

# ============================================================================
# Save Final Model
# ============================================================================

print("\nSaving final model...")

model.save(SAVEDMODEL_DIR)
model.save(H5_MODEL_PATH)

with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(class_map, f, indent=2)

print(f"Saved: {SAVEDMODEL_DIR}/")
print(f"Saved: {H5_MODEL_PATH}")
print(f"Saved: {CLASS_MAP_PATH}")

# ============================================================================
# Final Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

val_loss, val_acc = model.evaluate(val_generator, verbose=0)

print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Per-class accuracy
val_generator.reset()
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nPer-Class Accuracy:")
for class_idx in range(NUM_CLASSES):
    class_name = class_map[class_idx]
    mask = y_true == class_idx
    if mask.sum() > 0:
        acc = (y_pred[mask] == class_idx).mean()
        print(f"  {class_name:12s}: {acc*100:.2f}%")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Resumed)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy (Resumed)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_resumed.png', dpi=100, bbox_inches='tight')
print(f"\nSaved: training_history_resumed.png")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal Accuracy: {val_acc*100:.2f}%")
print("Model ready for deployment!")
