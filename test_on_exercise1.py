"""
Test FAISS Vector Similarity System on Exercise 1 Dataset
Evaluates the embedding model on unseen test data
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import faiss
import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL_PATH = "viewpoint_embedder.h5"
FAISS_INDEX_DIR = "faiss_index"
TEST_DIR = "exercise_1"  # Test dataset
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.7  # Cosine similarity threshold

print("=" * 70)
print("TESTING ON EXERCISE_1 DATASET")
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

print(f"✅ Loaded {len(class_names)} classes: {class_names}")

# ============================================================================
# 2. Collect Test Images
# ============================================================================

print(f"\n📂 Collecting test images from {TEST_DIR}...")

test_images = []
test_labels = []

test_path = Path(TEST_DIR)

# Recursively find all images
for class_name in class_names:
    class_dir = test_path / class_name
    if not class_dir.exists():
        print(f"  ⚠️  {class_name}: directory not found, skipping")
        continue
    
    images = list(class_dir.glob('**/*.jpg')) + list(class_dir.glob('**/*.png'))
    
    for img_path in images:
        test_images.append(str(img_path))
        test_labels.append(class_name)
    
    print(f"  {class_name}: {len(images)} test images")

print(f"\n✅ Total test images: {len(test_images)}")

if len(test_images) == 0:
    print("\n❌ No test images found!")
    print(f"   Make sure {TEST_DIR} contains subdirectories for each class")
    exit(1)

# ============================================================================
# 3. Inference Functions
# ============================================================================

def load_and_preprocess_image(img_path):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_viewpoint(img_path):
    """Predict viewpoint using FAISS + prototype distance"""
    # Extract embedding
    image = load_and_preprocess_image(img_path)
    query_embedding = embedding_model.predict(image, verbose=0).astype('float32')
    
    # Distance to prototypes (cosine similarity)
    prototype_similarities = np.dot(query_embedding, prototypes.T)[0]
    
    # Best match
    best_idx = np.argmax(prototype_similarities)
    best_similarity = prototype_similarities[best_idx]
    best_class = class_names[best_idx]
    
    # Apply threshold
    if best_similarity < UNKNOWN_THRESHOLD:
        predicted_class = 'unknown'
        confidence = best_similarity
    else:
        predicted_class = best_class
        confidence = best_similarity
    
    return predicted_class, confidence, prototype_similarities

# ============================================================================
# 4. Run Evaluation
# ============================================================================

print("\n🔄 Running evaluation...")

predictions = []
confidences = []
correct = 0
total = 0

# Per-class accuracy
class_correct = defaultdict(int)
class_total = defaultdict(int)
confusion_matrix = defaultdict(lambda: defaultdict(int))

for img_path, true_label in tqdm(zip(test_images, test_labels), total=len(test_images), desc="Testing"):
    try:
        pred_class, confidence, _ = predict_viewpoint(img_path)
        
        predictions.append(pred_class)
        confidences.append(confidence)
        
        # Update counts
        class_total[true_label] += 1
        total += 1
        
        if pred_class == true_label:
            correct += 1
            class_correct[true_label] += 1
        
        # Confusion matrix
        confusion_matrix[true_label][pred_class] += 1
        
    except Exception as e:
        print(f"\n⚠️  Error processing {img_path}: {e}")
        continue

# ============================================================================
# 5. Results
# ============================================================================

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

overall_accuracy = (correct / total) * 100 if total > 0 else 0

print(f"\n📊 Overall Performance:")
print(f"  Total images: {total}")
print(f"  Correct predictions: {correct}")
print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
print(f"  Mean Confidence: {np.mean(confidences):.4f}")

print(f"\n📈 Per-Class Accuracy:")
print("-" * 70)
for class_name in sorted(class_names):
    if class_total[class_name] > 0:
        acc = (class_correct[class_name] / class_total[class_name]) * 100
        print(f"  {class_name:12s}: {acc:6.2f}% ({class_correct[class_name]:4d}/{class_total[class_name]:4d})")
    else:
        print(f"  {class_name:12s}: No test samples")

print(f"\n🔀 Confusion Matrix:")
print("-" * 70)
print(f"{'True \\ Pred':12s} | " + " | ".join([f"{cn:8s}" for cn in class_names]))
print("-" * (13 + len(class_names) * 11))

for true_class in class_names:
    row = [f"{true_class:12s}"]
    for pred_class in class_names:
        count = confusion_matrix[true_class][pred_class]
        row.append(f"{count:8d}")
    print(" | ".join(row))

# Save results
results = {
    'overall_accuracy': overall_accuracy,
    'total_images': total,
    'correct': correct,
    'mean_confidence': float(np.mean(confidences)),
    'per_class_accuracy': {cn: (class_correct[cn] / class_total[cn] * 100 if class_total[cn] > 0 else 0)
                           for cn in class_names},
    'confusion_matrix': {tc: dict(confusion_matrix[tc]) for tc in class_names}
}

with open('test_results_exercise1.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: test_results_exercise1.json")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"\n✅ Test Accuracy: {overall_accuracy:.2f}%")
print(f"✅ Threshold: {UNKNOWN_THRESHOLD}")
