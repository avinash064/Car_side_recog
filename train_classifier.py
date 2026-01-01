"""
Train a 6-class car viewpoint classifier for mobile deployment.
Uses MobileNetV2 with mild augmentations (NO horizontal flip).
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = r"C:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog\cfv_viewpoint_train"
OUTPUT_DIR = r"C:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog\models"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Classes will be auto-detected from the dataset
# Expected: front, frontleft, frontright, rear, rearleft, rearright, unknown


def create_augmentation_layer():
    """
    Create augmentation layer WITHOUT horizontal flip.
    Left/right semantics must be preserved.
    """
    return keras.Sequential([
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name='augmentation')


def create_model(num_classes=6):
    """
    Create MobileNetV2-based classifier.
    Optimized for mobile deployment.
    """
    # Load pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Unfreeze last 30 layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = create_augmentation_layer()(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def main():
    print("=" * 60)
    print("Car Viewpoint Classifier Training")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading data from: {DATA_DIR}")
    train_ds = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    # Verify class order
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nDetected {num_classes} classes: {class_names}")
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    # Create model
    print("\nBuilding MobileNetV2 model...")
    model = create_model(num_classes=num_classes)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
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
        )
    ]
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\nSaving models...")
    
    # Keras format
    keras_path = os.path.join(OUTPUT_DIR, 'viewpoint_classifier.keras')
    model.save(keras_path)
    print(f"✓ Saved Keras model: {keras_path}")
    
    # SavedModel format (for TFLite conversion)
    savedmodel_path = os.path.join(OUTPUT_DIR, 'saved_model')
    model.save(savedmodel_path, save_format='tf')
    print(f"✓ Saved SavedModel: {savedmodel_path}")
    
    # Save class mapping
    class_map_path = os.path.join(OUTPUT_DIR, 'class_mapping.txt')
    with open(class_map_path, 'w') as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx}: {name}\n")
    print(f"✓ Saved class mapping: {class_map_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\n✓ Training complete!")
    print(f"Models saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
