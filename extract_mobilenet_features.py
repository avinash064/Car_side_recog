"""
MobileNet Feature Extractor for Vehicle Viewpoint Search
Uses pre-trained MobileNet (ImageNet) without any fine-tuning
Similar to DeepImageSearch approach - pure feature extraction + FAISS
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from pathlib import Path
import json
import faiss
from tqdm import tqdm
from PIL import Image

# Configuration
DATASET_DIR = "cfv_viewpoint_train"
IMG_SIZE = 224
OUTPUT_DIR = "faiss_features"
FEATURE_LAYER = 'block_13_expand_relu'  # Mid-level features from MobileNet

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("MOBILENET FEATURE EXTRACTION (NO TRAINING)")
print("=" * 70)
print("Using pre-trained MobileNet for deep image search")
print("=" * 70)

# ============================================================================
# 1. Build Feature Extractor
# ============================================================================

print("\nBuilding feature extractor...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Extract features from intermediate layer + GlobalAveragePooling
x = base_model.get_layer(FEATURE_LAYER).output
x = keras.layers.GlobalAveragePooling2D()(x)  # Reduce dimension

feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=x,
    name='mobilenet_feature_extractor'
)

# Test feature extraction
test_img = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
test_img = preprocess_input(test_img)
test_features = feature_extractor.predict(test_img, verbose=0)

print(f"[OK] Feature extractor ready")
print(f"[OK] Feature shape: {test_features.shape}")
print(f"[OK] Feature dimension: {test_features.shape[1]} (GlobalAveragePooling2D)")

# Feature dimension
feature_dim = test_features.shape[1]

# Save feature extractor
feature_extractor.save('mobilenet_feature_extractor.h5')
print("[OK] Saved: mobilenet_feature_extractor.h5")

# ============================================================================
# 2. Collect Image Paths
# ============================================================================

print(f"\n Collecting images from {DATASET_DIR}...")

image_paths = []
labels = []

dataset_path = Path(DATASET_DIR)
class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

for class_idx, class_name in enumerate(class_names):
    class_dir = dataset_path / class_name
    class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    
    for img_path in class_images:
        image_paths.append(str(img_path))
        labels.append(class_idx)
    
    print(f"  {class_name}: {len(class_images)} images")

print(f"\n[OK] Total images: {len(image_paths)}")
print(f"[OK] Classes: {class_names}")

# Save metadata
with open(os.path.join(OUTPUT_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'image_paths.json'), 'w') as f:
    json.dump(image_paths, f, indent=2)

# ============================================================================
# 3. Extract Features
# ============================================================================

print("\n Extracting features...")

def load_and_preprocess_image(img_path):
    """Load and preprocess image for MobileNet"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # MobileNet preprocessing
    return img_array

features_list = []
batch_size = 32

for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
    batch_paths = image_paths[i:i+batch_size]
    batch_images = np.vstack([load_and_preprocess_image(p) for p in batch_paths])
    
    # Extract features (already 1D from GlobalAveragePooling2D)
    batch_features = feature_extractor.predict(batch_images, verbose=0)
    
    features_list.append(batch_features)

features = np.vstack(features_list).astype('float32')
labels = np.array(labels, dtype=np.int32)

print(f"\n[OK] Features shape: {features.shape}")
print(f"[OK] Feature dimension: {features.shape[1]}")

# L2 normalize features for cosine similarity
features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
print(f"[OK] Features L2-normalized")

# ============================================================================
# 4. Compute Class Prototypes
# ============================================================================

print("\n Computing class prototypes...")

prototypes = {}
for class_idx, class_name in enumerate(class_names):
    class_mask = (labels == class_idx)
    class_features = features_normalized[class_mask]
    
    # Mean prototype
    prototype = class_features.mean(axis=0)
    # Re-normalize
    prototype = prototype / np.linalg.norm(prototype)
    
    prototypes[class_name] = prototype
    print(f"  {class_name}: {class_features.shape[0]} samples -> prototype")

# Save prototypes
prototypes_array = np.array([prototypes[cn] for cn in class_names]).astype('float32')
np.save(os.path.join(OUTPUT_DIR, 'prototypes.npy'), prototypes_array)
print(f"\n[OK] Saved prototypes: {prototypes_array.shape}")

# ============================================================================
# 5. Build FAISS Index
# ============================================================================

print("\n Building FAISS index...")

# Use Inner Product for cosine similarity (features are normalized)
index = faiss.IndexFlatIP(features.shape[1])
index.add(features_normalized)

print(f"[OK] FAISS index built: {index.ntotal} vectors")

# Save FAISS index
faiss.write_index(index, os.path.join(OUTPUT_DIR, 'faiss_index.bin'))
print(f"[OK] Saved FAISS index")

# ============================================================================
# 6. Save Features and Labels
# ============================================================================

print("\n Saving features and labels...")

np.save(os.path.join(OUTPUT_DIR, 'features.npy'), features_normalized)
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), labels)

print("[OK] Saved features and labels")

# ============================================================================
# 7. Test Search
# ============================================================================

print("\n Testing FAISS search...")

# Test with first image
test_feature = features_normalized[0:1]
test_label = labels[0]

k = 5
distances, indices = index.search(test_feature, k)

print(f"\nTest query: {image_paths[0]}")
print(f"True class: {class_names[test_label]}")
print(f"\nTop-{k} nearest neighbors:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor_label = labels[idx]
    neighbor_class = class_names[neighbor_label]
    print(f"  {i+1}. Similarity: {dist:.4f} | Class: {neighbor_class}")

# ============================================================================
# 8. Summary
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(f"\n Statistics:")
print(f"  Total images: {len(features):,}")
print(f"  Feature dimension: {features.shape[1]:,}")
print(f"  Number of classes: {len(class_names)}")
print(f"  FAISS index type: IndexFlatIP (cosine similarity)")
print(f"  Feature layer: {FEATURE_LAYER}")

print(f"\n Generated Files (in {OUTPUT_DIR}/):")
print("  - mobilenet_feature_extractor.h5  (Feature extractor model)")
print("  - faiss_index.bin                 (FAISS index)")
print("  - features.npy                    (L2-normalized features)")
print("  - labels.npy                      (Class labels)")
print("  - prototypes.npy                  (Class prototypes)")
print("  - class_names.json                (Class names)")
print("  - image_paths.json                (Image paths)")

print("\n Next Steps:")
print("  1. Use inference_mobilenet.py for similarity search")
print("  2. Test on exercise_1 with test_mobilenet.py")
print("  3. No training needed!")

print("\n[OK] Ready for deep image search!")
