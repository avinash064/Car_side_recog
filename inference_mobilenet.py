"""
MobileNet-based Viewpoint Inference (No Training Required)
Uses pre-trained MobileNet features + FAISS for similarity search
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import faiss
import json
from PIL import Image
import argparse

# Configuration
FEATURE_EXTRACTOR_PATH = "mobilenet_feature_extractor.h5"
FAISS_DIR = "faiss_features"
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.75  # Cosine similarity threshold

print("=" * 70)
print("MOBILENET-BASED VIEWPOINT SEARCH")
print("=" * 70)

# ============================================================================
# 1. Load Resources
# ============================================================================

print("\n📦 Loading resources...")

# Load feature extractor
feature_extractor = keras.models.load_model(FEATURE_EXTRACTOR_PATH)
print(f"✅ Feature extractor loaded")

# Get feature dimension
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
dummy_features = feature_extractor.predict(preprocess_input(dummy), verbose=0)
FEATURE_DIM = np.prod(dummy_features.shape[1:])

# Load FAISS index
faiss_index = faiss.read_index(os.path.join(FAISS_DIR, 'faiss_index.bin'))
print(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors")

# Load metadata
with open(os.path.join(FAISS_DIR, 'class_names.json'), 'r') as f:
    class_names = json.load(f)

labels = np.load(os.path.join(FAISS_DIR, 'labels.npy'))
prototypes = np.load(os.path.join(FAISS_DIR, 'prototypes.npy'))

print(f"✅ Loaded {len(class_names)} classes: {class_names}")

# ============================================================================
# 2. Inference Functions
# ============================================================================

def extract_features(img_path):
    """Extract MobileNet features from image"""
    # Load image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    features_flat = features.reshape(1, -1).astype('float32')
    
    # L2 normalize
    features_normalized = features_flat / np.linalg.norm(features_flat)
    
    return features_normalized

def predict_viewpoint(img_path, k=10, verbose=True):
    """
    Predict viewpoint using MobileNet features + FAISS
    
    Args:
        img_path: Path to image
        k: Number of nearest neighbors
        verbose: Print details
    
    Returns:
        dict with results
    """
    # Extract features
    query_features = extract_features(img_path)
    
    # Search FAISS
    distances, indices = faiss_index.search(query_features, k)
    
    # Get neighbor labels
    neighbor_labels = labels[indices[0]]
    neighbor_distances = distances[0]
    
    # Count votes
    class_votes = {cn: 0 for cn in class_names}
    class_distances = {cn: [] for cn in class_names}
    
    for label, dist in zip(neighbor_labels, neighbor_distances):
        cn = class_names[label]
        class_votes[cn] += 1
        class_distances[cn].append(dist)
    
    # Mean distance per class
    class_mean_distances = {}
    for cn in class_names:
        if class_distances[cn]:
            class_mean_distances[cn] = np.mean(class_distances[cn])
        else:
            class_mean_distances[cn] = -1.0
    
    # Distance to prototypes
    prototype_similarities = np.dot(query_features, prototypes.T)[0]
    
    best_proto_idx = np.argmax(prototype_similarities)
    best_proto_class = class_names[best_proto_idx]
    best_proto_sim = prototype_similarities[best_proto_idx]
    
    # Decision
    if best_proto_sim < UNKNOWN_THRESHOLD:
        predicted_class = 'unknown'
        confidence = best_proto_sim
        is_known = False
    else:
        predicted_class = best_proto_class
        confidence = best_proto_sim
        is_known = True
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'is_known': is_known,
        'prototype_similarities': {cn: float(prototype_similarities[i]) 
                                  for i, cn in enumerate(class_names)},
        'class_votes': class_votes,
        'class_mean_distances': class_mean_distances
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Query: {os.path.basename(img_path)}")
        print("=" * 70)
        
        print(f"\n🎯 PREDICTION: {predicted_class.upper()}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Status: {'✅ KNOWN' if is_known else '❌ UNKNOWN (rejected)'}")
        
        print(f"\n📐 Prototype Similarities:")
        for cn in sorted(class_names, key=lambda x: prototype_similarities[class_names.index(x)], reverse=True):
            idx = class_names.index(cn)
            sim = prototype_similarities[idx]
            marker = "✓" if cn == best_proto_class else " "
            print(f"  {marker} {cn:12s}: {sim:.4f}")
        
        print(f"\n📊 K-NN Votes (k={k}):")
        for cn in sorted(class_votes.keys(), key=lambda x: class_votes[x], reverse=True):
            votes = class_votes[cn]
            mean_dist = class_mean_distances[cn]
            print(f"  {cn:12s}: {votes:2d} votes | mean sim: {mean_dist:.4f}")
        
        print(f"\n⚙️  Decision:")
        print(f"  Threshold: {UNKNOWN_THRESHOLD:.4f}")
        print(f"  Best match: {best_proto_class} ({best_proto_sim:.4f})")
        if best_proto_sim < UNKNOWN_THRESHOLD:
            print(f"  → {best_proto_sim:.4f} < {UNKNOWN_THRESHOLD:.4f} → REJECT")
        else:
            print(f"  → {best_proto_sim:.4f} >= {UNKNOWN_THRESHOLD:.4f} → ACCEPT")
    
    return result

# ============================================================================
# 3. Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MobileNet-based viewpoint search')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--threshold', type=float, default=UNKNOWN_THRESHOLD, 
                        help='Unknown threshold')
    
    args = parser.parse_args()
    UNKNOWN_THRESHOLD = args.threshold
    
    result = predict_viewpoint(args.image, k=args.k, verbose=True)
    
    print("\n" + "=" * 70)
    print(f"✅ Final: {result['predicted_class'].upper()} ({result['confidence']:.4f})")
    print("=" * 70)
