"""
Vector Similarity Training for Vehicle Viewpoint Matching
Using MobileNetV2 + Embedding Layer + Metric Learning

This script trains an embedding model that maps images to a fixed-length vector space
where similar viewpoints are close together.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from pathlib import Path

# Configuration
DATASET_DIR = "cfv_viewpoint_train"  # Training dataset
TEST_DIR = "exercise_1"  # Test dataset (for later evaluation)
IMG_SIZE = 224
BATCH_SIZE = 32
EMBEDDING_DIM = 256  # 128 or 256
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Class names
CLASSES = ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']
NUM_CLASSES = len(CLASSES)

print("=" * 70)
print("VECTOR SIMILARITY TRAINING")
print("=" * 70)
print(f"Dataset: {DATASET_DIR}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Classes: {NUM_CLASSES}")
print("=" * 70)

# ============================================================================
# 1. Data Loading
# ============================================================================

print("\n📂 Loading dataset...")

# Mild augmentation (NO horizontal flip)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.9, 1.1],
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_indices = train_generator.class_indices
print(f"\n✅ Loaded {train_generator.samples} training images")
print(f"✅ Loaded {val_generator.samples} validation images")
print(f"\nClass mapping: {class_indices}")

# Save class mapping
with open('class_mapping.json', 'w') as f:
    json.dump(class_indices, f, indent=2)

# ============================================================================
# 2. Build Embedding Model
# ============================================================================

print("\n🏗️  Building embedding model...")

def build_embedding_model(embedding_dim=256):
    """
    Build model with:
    - MobileNetV2 backbone (ImageNet pretrained)
    - Global pooling
    - Embedding layer (normalized)
    - Temporary softmax head for training
    """
    # Base model
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    
    # Input
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Backbone
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Embedding layer (L2 normalized)
    embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)
    embeddings_normalized = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name='embeddings_normalized'
    )(embeddings)
    
    # Temporary softmax head for training
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='classifier')(embeddings_normalized)
    
    model = keras.Model(inputs, outputs, name='viewpoint_embedding_classifier')
    
    return model

model = build_embedding_model(EMBEDDING_DIM)
model.summary()

print(f"\n✅ Model built: {model.count_params():,} parameters")
print(f"✅ Embedding dimension: {EMBEDDING_DIM}")

# ============================================================================
# 3. Training Configuration
# ============================================================================

print("\n⚙️  Configuring training...")

# AdamW optimizer
optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'embedding_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.CSVLogger('training_history.csv')
]

# ============================================================================
# 4. Training
# ============================================================================

print("\n🚀 Starting training...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================================
# 5. Extract Embedding Model (Remove Softmax Head)
# ============================================================================

print("\n🔧 Extracting embedding model...")

# Create embedding-only model
embedding_model = keras.Model(
    inputs=model.input,
    outputs=model.get_layer('embeddings_normalized').output,
    name='viewpoint_embedder'
)

# Save embedding model
embedding_model.save('viewpoint_embedder.h5')
print("✅ Saved: viewpoint_embedder.h5")

# Save as SavedModel for TFLite conversion
embedding_model.save('viewpoint_embedder_savedmodel')
print("✅ Saved: viewpoint_embedder_savedmodel/")

# ============================================================================
# 6. Evaluation
# ============================================================================

print("\n📊 Final Evaluation:")
print("=" * 70)

# Evaluate classification accuracy (with softmax head)
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy (classification): {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Test embedding extraction
print("\n🧪 Testing embedding extraction...")
test_image = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
test_embedding = embedding_model.predict(test_image, verbose=0)
print(f"✅ Embedding shape: {test_embedding.shape}")
print(f"✅ Embedding norm: {np.linalg.norm(test_embedding):.4f} (should be ~1.0)")

# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print("\n📦 Generated Files:")
print("  - embedding_model_best.h5         (full model with softmax)")
print("  - viewpoint_embedder.h5           (embedding model only)")
print("  - viewpoint_embedder_savedmodel/  (for TFLite conversion)")
print("  - class_mapping.json              (class indices)")
print("  - training_history.csv            (training logs)")

print("\n🎯 Next Steps:")
print("  1. Run extract_embeddings_and_build_faiss.py to build FAISS index")
print("  2. Use inference_faiss.py for similarity-based predictions")
print("  3. Convert to TFLite for deployment")

print("\n✅ Ready for vector similarity matching!")
