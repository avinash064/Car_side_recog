"""
Fast Training Script for 7-Class Canonical Vehicle Viewpoint Classifier
========================================================================
Optimized for speed with reduced epochs and efficient settings.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ============================================================================
# OPTIMIZED Configuration for FAST Training
# ============================================================================

DATASET_DIR = "cfv_viewpoint_train"
IMG_SIZE = 224
BATCH_SIZE = 64  # Increased for faster training
NUM_CLASSES = 7

# 50 EPOCHS for thorough training
WARMUP_EPOCHS = 5      # Warmup with frozen backbone
FINETUNE_EPOCHS = 45   # Full fine-tuning (50 total epochs)
WARMUP_LR = 5e-4       # Slightly higher for faster convergence
FINETUNE_LR = 2e-4     # Slightly higher for faster convergence
WEIGHT_DECAY = 1e-4

# Output paths
SAVEDMODEL_DIR = "viewpoint_model_savedmodel"
H5_MODEL_PATH = "viewpoint_model.h5"
CLASS_MAP_PATH = "class_map.json"

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision for faster training (even on CPU)
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

print("=" * 80)
print("FAST CANONICAL VEHICLE VIEWPOINT CLASSIFIER TRAINING")
print("=" * 80)
print(f"\nOPTIMIZATION: Mixed precision enabled")
print(f"OPTIMIZATION: Batch size increased to {BATCH_SIZE}")
print(f"OPTIMIZATION: Total epochs reduced to {WARMUP_EPOCHS + FINETUNE_EPOCHS}")

# ============================================================================
# Data Loading
# ============================================================================

print("\n[1/6] Loading dataset...")

# Simplified augmentation for speed
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

print("\n[2/6] Computing class weights...")

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

# ============================================================================
# Build Model
# ============================================================================

print("\n[3/6] Building model...")

def build_model(trainable_backbone=False):
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

model = build_model(trainable_backbone=False)
print(f"Model built ({model.count_params():,} parameters)")

# ============================================================================
# Stage 1: Warm-up (Frozen Backbone)
# ============================================================================

print(f"\n[4/6] Stage 1: Warm-up ({WARMUP_EPOCHS} epochs)...")

optimizer = keras.optimizers.Adam(
    learning_rate=WARMUP_LR
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

warmup_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
]

history_warmup = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=WARMUP_EPOCHS,
    class_weight=class_weights,
    callbacks=warmup_callbacks,
    verbose=1
)

print("Warm-up completed")

# ============================================================================
# Stage 2: Fine-tuning (Unfrozen Backbone)
# ============================================================================

print(f"\n[5/6] Stage 2: Fine-tuning ({FINETUNE_EPOCHS} epochs)...")

model.layers[1].trainable = True

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
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_viewpoint_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
]

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINETUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=finetune_callbacks,
    verbose=1
)

print("Fine-tuning completed")

# ============================================================================
# Save Model
# ============================================================================

print("\n[6/6] Saving model...")

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
combined_history = {
    'loss': history_warmup.history['loss'] + history_finetune.history['loss'],
    'val_loss': history_warmup.history['val_loss'] + history_finetune.history['val_loss'],
    'accuracy': history_warmup.history['accuracy'] + history_finetune.history['accuracy'],
    'val_accuracy': history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(combined_history['loss'], label='Train', linewidth=2)
axes[0].plot(combined_history['val_loss'], label='Val', linewidth=2)
axes[0].axvline(x=WARMUP_EPOCHS-1, color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(combined_history['accuracy'], label='Train', linewidth=2)
axes[1].plot(combined_history['val_accuracy'], label='Val', linewidth=2)
axes[1].axvline(x=WARMUP_EPOCHS-1, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
print(f"\nSaved: training_history.png")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal Accuracy: {val_acc*100:.2f}%")
print(f"Total Epochs: {WARMUP_EPOCHS + FINETUNE_EPOCHS}")
print("\nModel ready for deployment!")
