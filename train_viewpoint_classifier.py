"""
Complete Training Script for 7-Class Canonical Vehicle Viewpoint Classifier
===========================================================================

Trains a MobileNetV2-based classifier to predict:
  front, frontleft, frontright, rear, rearleft, rearright, unknown

Requirements:
  - Dataset: cfv_viewpoint_train/ with 7 class subdirectories
  - MobileNetV2 backbone (ImageNet pretrained)
  - Two-stage training: frozen backbone warmup → full fine-tuning
  - Class weighting to prevent 'unknown' from dominating
  - Augmentation: brightness jitter, translation, zoom (NO horizontal flip)

Outputs:
  - viewpoint_model_savedmodel/ (TensorFlow SavedModel)
  - viewpoint_model.h5 (Keras H5 format)
  - class_map.json (class index mapping)
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
# Configuration
# ============================================================================

# Dataset path
DATASET_DIR = "cfv_viewpoint_train"

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Training configuration - Stage 1: Warm-up with frozen backbone
WARMUP_EPOCHS = 4
WARMUP_LR = 3e-4

# Training configuration - Stage 2: Full fine-tuning
FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Output paths
SAVEDMODEL_DIR = "viewpoint_model_savedmodel"
H5_MODEL_PATH = "viewpoint_model.h5"
CLASS_MAP_PATH = "class_map.json"

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("CANONICAL VEHICLE VIEWPOINT CLASSIFIER - TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. Data Loading and Preparation
# ============================================================================

print("\n[1/6] Loading dataset...")

# Define data augmentation - NO HORIZONTAL FLIP (preserves left/right semantics)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8, 1.2],      # Brightness jitter
    width_shift_range=0.1,             # Small horizontal translation
    height_shift_range=0.1,            # Small vertical translation
    zoom_range=0.1,                    # Small zoom
    fill_mode='nearest',
    validation_split=0.2               # 80/20 train/val split
)

# No augmentation for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Save class mapping
class_indices = train_generator.class_indices
class_map = {v: k for k, v in class_indices.items()}

print(f"\nDataset loaded successfully!")
print(f"  Total training samples: {train_generator.samples}")
print(f"  Total validation samples: {val_generator.samples}")
print(f"\nClass distribution:")
for class_name, class_idx in sorted(class_indices.items(), key=lambda x: x[1]):
    count = len(os.listdir(os.path.join(DATASET_DIR, class_name)))
    print(f"  {class_idx}: {class_name:12s} - {count:5d} images")

# ============================================================================
# 2. Compute Class Weights (prevent 'unknown' from dominating)
# ============================================================================

print("\n[2/6] Computing class weights...")

# Collect all class labels from training set
train_labels = []
for class_name, class_idx in class_indices.items():
    class_dir = os.path.join(DATASET_DIR, class_name)
    num_samples = len(os.listdir(class_dir))
    
    # Approximate: 80% for training (due to 80/20 split)
    train_samples = int(num_samples * 0.8)
    train_labels.extend([class_idx] * train_samples)

train_labels = np.array(train_labels)

# Compute balanced class weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

print("\nClass weights (to balance dataset):")
for idx, weight in class_weights.items():
    print(f"  {idx}: {class_map[idx]:12s} - weight: {weight:.3f}")

# ============================================================================
# 3. Build Model (MobileNetV2 + Custom Head)
# ============================================================================

print("\n[3/6] Building model...")

def build_model(trainable_backbone=False):
    """
    Build MobileNetV2-based classifier.
    
    Args:
        trainable_backbone: If True, backbone weights are trainable.
                          If False, backbone is frozen (for warmup).
    """
    # Load MobileNetV2 pretrained on ImageNet (exclude top classification layer)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze/unfreeze backbone
    base_model.trainable = trainable_backbone
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # MobileNetV2 backbone
    x = base_model(inputs, training=False)  # Set training=False for inference mode
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.2)(x)
    
    # Classification head
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='viewpoint_output')(x)
    
    model = keras.Model(inputs, outputs, name='viewpoint_classifier')
    
    return model

# Build initial model with frozen backbone
model = build_model(trainable_backbone=False)

print(f"\nModel architecture:")
print(f"  Backbone: MobileNetV2 (ImageNet pretrained)")
print(f"  Input size: {IMG_SIZE}x{IMG_SIZE}x3")
print(f"  Output: {NUM_CLASSES} classes (softmax)")
print(f"  Total parameters: {model.count_params():,}")

# ============================================================================
# 4. Stage 1: Warm-up Training (Frozen Backbone)
# ============================================================================

print("\n[4/6] Stage 1: Warm-up training with frozen backbone...")
print(f"  Epochs: {WARMUP_EPOCHS}")
print(f"  Learning rate: {WARMUP_LR}")

# Compile model with AdamW optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=WARMUP_LR,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
)

# Callbacks
warmup_callbacks = [
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
    )
]

# Train
history_warmup = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=WARMUP_EPOCHS,
    class_weight=class_weights,
    callbacks=warmup_callbacks,
    verbose=1
)

print("\nWarm-up training completed!")

# ============================================================================
# 5. Stage 2: Full Fine-tuning (Unfreeze Backbone)
# ============================================================================

print("\n[5/6] Stage 2: Fine-tuning with unfrozen backbone...")
print(f"  Epochs: {FINETUNE_EPOCHS}")
print(f"  Learning rate: {FINETUNE_LR}")

# Unfreeze the backbone
model.layers[1].trainable = True  # base_model is layer 1

# Recompile with lower learning rate
optimizer_finetune = keras.optimizers.AdamW(
    learning_rate=FINETUNE_LR,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer_finetune,
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
)

print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Callbacks
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

# Train
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINETUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=finetune_callbacks,
    verbose=1
)

print("\nFine-tuning completed!")

# ============================================================================
# 6. Save Model and Artifacts
# ============================================================================

print("\n[6/6] Saving model and artifacts...")

# Save as TensorFlow SavedModel
model.save(SAVEDMODEL_DIR)
print(f"  ✓ Saved TensorFlow SavedModel: {SAVEDMODEL_DIR}/")

# Save as Keras H5
model.save(H5_MODEL_PATH)
print(f"  ✓ Saved Keras H5 model: {H5_MODEL_PATH}")

# Save class mapping
with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(class_map, f, indent=2)
print(f"  ✓ Saved class mapping: {CLASS_MAP_PATH}")

# ============================================================================
# Final Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

# Evaluate on validation set
val_loss, val_acc, val_top2 = model.evaluate(val_generator, verbose=0)

print(f"\nValidation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_acc*100:.2f}%")
print(f"  Top-2 Accuracy: {val_top2*100:.2f}%")

# Per-class accuracy
print("\nComputing per-class metrics...")

# Get predictions
val_generator.reset()
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Per-class accuracy
print("\nPer-class Accuracy:")
for class_idx in range(NUM_CLASSES):
    class_name = class_map[class_idx]
    mask = y_true == class_idx
    if mask.sum() > 0:
        acc = (y_pred[mask] == class_idx).mean()
        print(f"  {class_name:12s}: {acc*100:.2f}% ({mask.sum()} samples)")

# ============================================================================
# Plot Training History
# ============================================================================

print("\nGenerating training plots...")

# Combine histories
combined_history = {
    'loss': history_warmup.history['loss'] + history_finetune.history['loss'],
    'val_loss': history_warmup.history['val_loss'] + history_finetune.history['val_loss'],
    'accuracy': history_warmup.history['accuracy'] + history_finetune.history['accuracy'],
    'val_accuracy': history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
axes[0].plot(combined_history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(combined_history['val_loss'], label='Val Loss', linewidth=2)
axes[0].axvline(x=WARMUP_EPOCHS-1, color='red', linestyle='--', label='Fine-tune Start')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot accuracy
axes[1].plot(combined_history['accuracy'], label='Train Acc', linewidth=2)
axes[1].plot(combined_history['val_accuracy'], label='Val Acc', linewidth=2)
axes[1].axvline(x=WARMUP_EPOCHS-1, color='red', linestyle='--', label='Fine-tune Start')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved training plot: training_history.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"""
✓ Model trained successfully!

Output Files:
  1. {SAVEDMODEL_DIR}/     - TensorFlow SavedModel (production deployment)
  2. {H5_MODEL_PATH}       - Keras H5 format (for further training)
  3. {CLASS_MAP_PATH}      - Class index mapping
  4. training_history.png  - Training curves visualization
  5. best_viewpoint_model.h5 - Best checkpoint during training

Model Specifications:
  • Architecture: MobileNetV2 (ImageNet pretrained)
  • Input: 224×224×3 RGB images
  • Output: 7 classes (front, frontleft, frontright, rear, rearleft, rearright, unknown)
  • Training: Two-stage (warmup + fine-tune)
  • Class balancing: Applied via class weights
  • Augmentation: Brightness, translation, zoom (NO horizontal flip)

Final Performance:
  • Validation Accuracy: {val_acc*100:.2f}%
  • Validation Top-2 Accuracy: {val_top2*100:.2f}%

Next Steps:
  1. Convert to TFLite for mobile deployment
  2. Test on unseen images
  3. Deploy to production

""")

print("=" * 80)
