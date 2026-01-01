"""
Batch Test on Exercise 1 Dataset
Processes all images in exercise_1, matches them using FAISS, and saves results
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import faiss
import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
import shutil
from tqdm import tqdm

# Configuration
FEATURE_EXTRACTOR_PATH = "mobilenet_feature_extractor.h5"
FAISS_DIR = "faiss_features"
TEST_DIR = "exercise_1"
OUTPUT_DIR = "exercise1_results"
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.75 

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("BATCH TESTING ON EXERCISE_1")
print("=" * 70)

# ============================================================================
# 1. Load Resources
# ============================================================================

print("\nLoading resources...")

# Load feature extractor
feature_extractor = keras.models.load_model(FEATURE_EXTRACTOR_PATH)
print("Feature extractor loaded")

# Load FAISS index
faiss_index = faiss.read_index(os.path.join(FAISS_DIR, 'faiss_index.bin'))
print(f"FAISS index loaded: {faiss_index.ntotal} vectors")

# Load metadata
with open(os.path.join(FAISS_DIR, 'class_names.json'), 'r') as f:
    class_names = json.load(f)

labels = np.load(os.path.join(FAISS_DIR, 'labels.npy'))
prototypes = np.load(os.path.join(FAISS_DIR, 'prototypes.npy'))

print(f"Classes: {class_names}\n")

# ============================================================================
# 2. Collect Test Images
# ============================================================================

print(f"Collecting images from {TEST_DIR}...")

test_data = []  # List of (image_path, subfolder, true_folder)

test_path = Path(TEST_DIR)

for subfolder in test_path.iterdir():
    if not subfolder.is_dir():
        continue
    
    # Recursively find all images
    images = list(subfolder.glob('**/*.jpg')) + list(subfolder.glob('**/*.png'))
    
    for img_path in images:
        # Determine true label from folder structure
        # If structure is exercise_1/subfolder/class_name/image.jpg
        parent_folder = img_path.parent.name
        
        test_data.append({
            'path': str(img_path),
            'subfolder': subfolder.name,
            'parent_folder': parent_folder,
            'filename': img_path.name
        })
    
    print(f"  {subfolder.name}: {len(images)} images")

print(f"\nTotal test images: {len(test_data)}")

if len(test_data) == 0:
    print("\nNo images found!")
    exit(1)

# ============================================================================
# 3. Extract Features and inference
# ============================================================================

def extract_features(img_path):
    """Extract MobileNet features"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = feature_extractor.predict(img_array, verbose=0)
    features_flat = features.reshape(1, -1).astype('float32')
    features_normalized = features_flat / np.linalg.norm(features_flat)
    
    return features_normalized

print("\nProcessing images...")

results = []

for item in tqdm(test_data, desc="Testing"):
    try:
        # Extract features
        query_features = extract_features(item['path'])
        
        # Distance to prototypes
        prototype_similarities = np.dot(query_features, prototypes.T)[0]
        
        best_idx = np.argmax(prototype_similarities)
        best_sim = prototype_similarities[best_idx]
        best_class = class_names[best_idx]
        
        # Decision
        if best_sim < UNKNOWN_THRESHOLD:
            predicted_class = 'unknown'
            confidence = best_sim
        else:
            predicted_class = best_class
            confidence = best_sim
        
        # Store result
        result = {
            'filename': item['filename'],
            'path': item['path'],
            'subfolder': item['subfolder'],
            'parent_folder': item['parent_folder'],
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'best_match': best_class,
            'best_similarity': float(best_sim),
            'all_similarities': {cn: float(prototype_similarities[i]) 
                                for i, cn in enumerate(class_names)}
        }
        
        results.append(result)
        
    except Exception as e:
        print(f"\nError processing {item['path']}: {e}")
        continue

# ============================================================================
# 4. Save Results
# ============================================================================

print(f"\nSaving results to {OUTPUT_DIR}/...")

# Save JSON results
with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Saved: predictions.json")

# Create organized output folders
for class_name in class_names + ['unknown']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'matched', class_name), exist_ok=True)

# Copy images to matched folders
print("\nOrganizing matched images...")

for result in tqdm(results, desc="Copying"):
    predicted = result['predicted_class']
    src_path = result['path']
    dst_path = os.path.join(OUTPUT_DIR, 'matched', predicted, result['filename'])
    
    # If duplicate filename, add subfolder prefix
    if os.path.exists(dst_path):
        base, ext = os.path.splitext(result['filename'])
        dst_path = os.path.join(OUTPUT_DIR, 'matched', predicted, 
                                f"{result['subfolder']}_{base}{ext}")
    
    shutil.copy2(src_path, dst_path)

print(f"Images organized in: {OUTPUT_DIR}/matched/")

# ============================================================================
# 5. Generate Statistics
# ============================================================================

print("\nGenerating statistics...")

# Count predictions
prediction_counts = defaultdict(int)
confidence_by_class = defaultdict(list)

for result in results:
    pred = result['predicted_class']
    conf = result['confidence']
    prediction_counts[pred] += 1
    confidence_by_class[pred].append(conf)

# Summary report
report = {
    'total_images': len(results),
    'threshold': UNKNOWN_THRESHOLD,
    'prediction_distribution': dict(prediction_counts),
    'mean_confidence_by_class': {
        cn: float(np.mean(confidence_by_class[cn])) if confidence_by_class[cn] else 0.0
        for cn in class_names + ['unknown']
    },
    'class_names': class_names
 }

with open(os.path.join(OUTPUT_DIR, 'summary_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print("Saved: summary_report.json")

# ============================================================================
# 6. Print Summary
# ============================================================================

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)

print(f"\nTotal images processed: {len(results)}")
print(f"Threshold: {UNKNOWN_THRESHOLD}")

print("\nPrediction Distribution:")
for class_name in sorted(prediction_counts.keys(), key=lambda x: prediction_counts[x], reverse=True):
    count = prediction_counts[class_name]
    pct = (count / len(results)) * 100
    mean_conf = report['mean_confidence_by_class'][class_name]
    print(f"  {class_name:12s}: {count:4d} ({pct:5.1f}%) | Avg confidence: {mean_conf:.4f}")

print(f"\nOutput files in: {OUTPUT_DIR}/")
print("  - predictions.json           (All predictions)")
print("  - summary_report.json        (Statistics)")
print("  - matched/[class_name]/      (Organized images)")

print("\nDone!")
