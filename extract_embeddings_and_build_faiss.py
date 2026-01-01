"""
Extract Embeddings and Build FAISS Index
Generates vector embeddings for all training images and creates FAISS index for fast similarity search
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import faiss
from tqdm import tqdm
from PIL import Image

# Configuration
DATASET_DIR = "cfv_viewpoint_train"
EMBEDDING_MODEL_PATH = "viewpoint_embedder.h5"
IMG_SIZE = 224
OUTPUT_DIR = "faiss_index"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FAISS INDEX CONSTRUCTION")
print("=" * 70)

# ============================================================================
# 1. Load Embedding Model
# ============================================================================

print("\n📦 Loading embedding model...")
embedding_model = keras.models.load_model(EMBEDDING_MODEL_PATH)
print(f"✅ Model loaded: {EMBEDDING_MODEL_PATH}")

# Get embedding dimension
dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
dummy_embedding = embedding_model.predict(dummy_input, verbose=0)
EMBEDDING_DIM = dummy_embedding.shape[1]
print(f"✅ Embedding dimension: {EMBEDDING_DIM}")

# ============================================================================
# 2. Collect All Image Paths
# ============================================================================

print("\n📂 Collecting image paths...")

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

print(f"\n✅ Total images: {len(image_paths)}")
print(f"✅ Classes: {class_names}")

# Save class names
with open(os.path.join(OUTPUT_DIR, 'class_names.json'), 'w') as f:
    json.dump(class_names, f, indent=2)

# ============================================================================
# 3. Extract Embeddings
# ============================================================================

print("\n🔄 Extracting embeddings...")

def load_and_preprocess_image(img_path):
    """Load and preprocess image for embedding extraction"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

embeddings = []
batch_size = 32

for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
    batch_paths = image_paths[i:i+batch_size]
    batch_images = np.vstack([load_and_preprocess_image(p) for p in batch_paths])
    batch_embeddings = embedding_model.predict(batch_images, verbose=0)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings).astype('float32')
labels = np.array(labels, dtype=np.int32)

print(f"\n✅ Embeddings shape: {embeddings.shape}")
print(f"✅ Labels shape: {labels.shape}")

# Verify L2 normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f"✅ Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

# ============================================================================
# 4. Compute Class Prototypes
# ============================================================================

print("\n📐 Computing class prototypes...")

prototypes = {}
for class_idx, class_name in enumerate(class_names):
    class_mask = (labels == class_idx)
    class_embeddings = embeddings[class_mask]
    
    # Mean of normalized vectors
    prototype = class_embeddings.mean(axis=0)
    # Re-normalize
    prototype = prototype / np.linalg.norm(prototype)
    
    prototypes[class_name] = prototype
    print(f"  {class_name}: {class_embeddings.shape[0]} samples → prototype")

# Save prototypes
prototypes_array = np.array([prototypes[cn] for cn in class_names]).astype('float32')
np.save(os.path.join(OUTPUT_DIR, 'prototypes.npy'), prototypes_array)
print(f"\n✅ Saved prototypes: {prototypes_array.shape}")

# ============================================================================
# 5. Build FAISS Index
# ============================================================================

print("\n🔍 Building FAISS index...")

# Use Inner Product (equivalent to cosine similarity for normalized vectors)
index = faiss.IndexFlatIP(EMBEDDING_DIM)

# Add embeddings to index
index.add(embeddings)

print(f"✅ FAISS index built: {index.ntotal} vectors")

# Save FAISS index
faiss.write_index(index, os.path.join(OUTPUT_DIR, 'faiss_index.bin'))
print(f"✅ Saved FAISS index")

# ============================================================================
# 6. Save Metadata
# ============================================================================

print("\n💾 Saving metadata...")

# Save embeddings
np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), labels)

# Save image paths
with open(os.path.join(OUTPUT_DIR, 'image_paths.json'), 'w') as f:
    json.dump(image_paths, f, indent=2)

print("✅ Saved embeddings, labels, and image paths")

# ============================================================================
# 7. Test FAISS Search
# ============================================================================

print("\n🧪 Testing FAISS search...")

# Test with first image
test_embedding = embeddings[0:1]
test_label = labels[0]

# Search for top-5 nearest neighbors
k = 5
distances, indices = index.search(test_embedding, k)

print(f"\nTest query: {image_paths[0]}")
print(f"True class: {class_names[test_label]}")
print(f"\nTop-{k} nearest neighbors:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    neighbor_label = labels[idx]
    neighbor_class = class_names[neighbor_label]
    print(f"  {i+1}. Distance: {dist:.4f} | Class: {neighbor_class} | Path: {os.path.basename(image_paths[idx])}")

# ============================================================================
# 8. Summary
# ============================================================================

print("\n" + "=" * 70)
print("FAISS INDEX CONSTRUCTION COMPLETE")
print("=" * 70)

print(f"\n📊 Statistics:")
print(f"  Total embeddings: {len(embeddings):,}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")
print(f"  Number of classes: {len(class_names)}")
print(f"  FAISS index type: IndexFlatIP (cosine similarity)")

print(f"\n📦 Generated Files (in {OUTPUT_DIR}/):")
print("  - faiss_index.bin       (FAISS index)")
print("  - embeddings.npy        (all embeddings)")
print("  - labels.npy            (class labels)")
print("  - prototypes.npy        (class prototype vectors)")
print("  - class_names.json      (class name mapping)")
print("  - image_paths.json      (image file paths)")

print("\n🎯 Next Step:")
print("  Use inference_faiss.py for similarity-based predictions")

print("\n✅ Ready for deployment!")
