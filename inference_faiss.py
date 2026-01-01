"""
FAISS-based Inference for Vehicle Viewpoint Matching
Uses vector similarity + distance thresholding for robust canonical viewpoint matching
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import faiss
import json
from PIL import Image
import argparse

# Configuration
EMBEDDING_MODEL_PATH = "viewpoint_embedder.h5"
FAISS_INDEX_DIR = "faiss_index"
IMG_SIZE = 224

# Distance threshold for "unknown" rejection
UNKNOWN_THRESHOLD = 0.7  # Cosine similarity threshold (0-1, higher is more similar)

print("=" * 70)
print("FAISS-BASED VIEWPOINT INFERENCE")
print("=" * 70)

# ============================================================================
# 1. Load Resources
# ============================================================================

print("\n📦 Loading resources...")

# Load embedding model
embedding_model = keras.models.load_model(EMBEDDING_MODEL_PATH)
print(f"✅ Embedding model loaded")

# Load FAISS index
faiss_index = faiss.read_index(os.path.join(FAISS_INDEX_DIR, 'faiss_index.bin'))
print(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors")

# Load metadata
with open(os.path.join(FAISS_INDEX_DIR, 'class_names.json'), 'r') as f:
    class_names = json.load(f)

labels = np.load(os.path.join(FAISS_INDEX_DIR, 'labels.npy'))
prototypes = np.load(os.path.join(FAISS_INDEX_DIR, 'prototypes.npy'))

print(f"✅ Loaded {len(class_names)} class prototypes")
print(f"✅ Classes: {class_names}")

# ============================================================================
# 2. Inference Functions
# ============================================================================

def load_and_preprocess_image(img_path):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_embedding(image):
    """Extract normalized embedding from image"""
    embedding = embedding_model.predict(image, verbose=0)
    return embedding.astype('float32')

def predict_viewpoint(img_path, k=10, verbose=True):
    """
    Predict viewpoint using FAISS search + distance thresholding
    
    Args:
        img_path: Path to image file
        k: Number of nearest neighbors to retrieve
        verbose: Print detailed results
    
    Returns:
        dict with prediction results
    """
    # Load and extract embedding
    image = load_and_preprocess_image(img_path)
    query_embedding = extract_embedding(image)
    
    # Search FAISS index (Inner Product = cosine similarity for normalized vectors)
    distances, indices = faiss_index.search(query_embedding, k)
    
    # Get neighbor labels
    neighbor_labels = labels[indices[0]]
    neighbor_distances = distances[0]
    
    # Count votes per class
    class_votes = {cn: 0 for cn in class_names}
    class_distances = {cn: [] for cn in class_names}
    
    for label, dist in zip(neighbor_labels, neighbor_distances):
        class_name = class_names[label]
        class_votes[class_name] += 1
        class_distances[class_name].append(dist)
    
    # Compute mean distance to each class
    class_mean_distances = {}
    for cn in class_names:
        if class_distances[cn]:
            class_mean_distances[cn] = np.mean(class_distances[cn])
        else:
            class_mean_distances[cn] = -1.0  # Very dissimilar
    
    # Get best match
    best_class = max(class_mean_distances, key=class_mean_distances.get)
    best_distance = class_mean_distances[best_class]
    
    # Distance to prototype
    prototype_distances = {}
    for idx, cn in enumerate(class_names):
        prototype = prototypes[idx:idx+1]
        # Cosine similarity (inner product for normalized vectors)
        similarity = np.dot(query_embedding, prototype.T)[0, 0]
        prototype_distances[cn] = similarity
    
    best_prototype_class = max(prototype_distances, key=prototype_distances.get)
    best_prototype_distance = prototype_distances[best_prototype_class]
    
    # Final decision: use prototype distance with threshold
    if best_prototype_distance < UNKNOWN_THRESHOLD:
        predicted_class = 'unknown'
        confidence = 1.0 - best_prototype_distance  # Distance to unknown
    else:
        predicted_class = best_prototype_class
        confidence = best_prototype_distance
    
    # Results
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'best_match': best_class,
        'best_match_distance': best_distance,
        'prototype_match': best_prototype_class,
        'prototype_distance': best_prototype_distance,
        'is_known': best_prototype_distance >= UNKNOWN_THRESHOLD,
        'class_votes': class_votes,
        'class_mean_distances': class_mean_distances,
        'prototype_distances': prototype_distances
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Query Image: {os.path.basename(img_path)}")
        print("=" * 70)
        
        print(f"\n🎯 PREDICTION: {predicted_class.upper()}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Status: {'✅ KNOWN' if result['is_known'] else '❌ UNKNOWN (rejected)'}")
        
        print(f"\n📊 K-NN Analysis (k={k}):")
        for cn in sorted(class_votes.keys(), key=lambda x: class_votes[x], reverse=True):
            votes = class_votes[cn]
            mean_dist = class_mean_distances[cn]
            print(f"  {cn:12s}: {votes:2d} votes | avg. distance: {mean_dist:.4f}")
        
        print(f"\n📐 Prototype Distances:")
        for cn in sorted(prototype_distances.keys(), key=lambda x: prototype_distances[x], reverse=True):
            dist = prototype_distances[cn]
            marker = "✓" if cn == best_prototype_class else " "
            print(f"  {marker} {cn:12s}: {dist:.4f}")
        
        print(f"\n⚙️  Decision Rule:")
        print(f"  Threshold: {UNKNOWN_THRESHOLD:.4f}")
        print(f"  Best prototype match: {best_prototype_class} ({best_prototype_distance:.4f})")
        
        if best_prototype_distance < UNKNOWN_THRESHOLD:
            print(f"  → {best_prototype_distance:.4f} < {UNKNOWN_THRESHOLD:.4f} → REJECT as 'unknown'")
        else:
            print(f"  → {best_prototype_distance:.4f} >= {UNKNOWN_THRESHOLD:.4f} → ACCEPT as '{predicted_class}'")
    
    return result

# ============================================================================
#  3. Main Inference
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FAISS-based viewpoint inference')
    parser.add_argument('--image', type=str, required=True, help='Path to query image')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors')
    parser.add_argument('--threshold', type=float, default=UNKNOWN_THRESHOLD, 
                        help='Unknown detection threshold')
    
    args = parser.parse_args()
    
    # Update threshold
    UNKNOWN_THRESHOLD = args.threshold
    
    # Run inference
    result = predict_viewpoint(args.image, k=args.k, verbose=True)
    
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"\n✅ Final Prediction: {result['predicted_class'].upper()}")
    print(f"✅ Confidence: {result['confidence']:.4f}")
    
    # Example usage in code:
    print("\n\n💡 Usage in Code:")
    print("```python")
    print("from inference_faiss import predict_viewpoint")
    print("")
    print("result = predict_viewpoint('path/to/car.jpg', k=10, verbose=False)")
    print("print(f\"Predicted: {result['predicted_class']}\")")
    print("print(f\"Confidence: {result['confidence']:.4f}\")")
    print("print(f\"Is known viewpoint: {result['is_known']}\")")
    print("```")
