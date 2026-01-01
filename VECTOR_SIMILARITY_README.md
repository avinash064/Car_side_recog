# Vector Similarity System with FAISS - Complete Workflow

## 🎯 System Overview

This is a **vector similarity system** for canonical viewpoint matching:
- Train on `cfv_viewpoint_train` (reference templates)
- Build FAISS index for fast similarity search
- Test on `exercise_1` (unseen test data)
- Reject unknown/invalid views using distance thresholding

---

## 📁 Dataset Structure

```
cfv_viewpoint_train/          ← Training dataset (canonical templates)
 ├── front/
 ├── frontleft/
 ├── frontright/
 ├── rear/
 ├── rearleft/
 ├── rearright/
 └── unknown/

exercise_1/                    ← Test dataset (validation)
 └── (same structure)
```

---

## 🚀 Complete Workflow

### Step 1: Train Embedding Model

```bash
python train_embedding_model.py
```

**What it does:**
- Trains MobileNetV2 with L2-normalized 256-dim embeddings
- Uses AdamW optimizer (lr=3e-4, weight_decay=1e-4)
- Mild augmentation (NO horizontal flip)
- Saves embedding model (without softmax head)

**Output files:**
- `viewpoint_embedder.h5` - Embedding model
- `viewpoint_embedder_savedmodel/` - For TFLite conversion
- `class_mapping.json` - Class indices

---

### Step 2: Extract Embeddings & Build FAISS Index

```bash
python extract_embeddings_and_build_faiss.py
```

**What it does:**
- Extracts embeddings for all training images
- Computes class prototype vectors (mean per class)
- Builds FAISS index (IndexFlatIP for cosine similarity)
- Saves everything in `faiss_index/`

**Output files:**
- `faiss_index/faiss_index.bin` - FAISS index
- `faiss_index/embeddings.npy` - All training embeddings
- `faiss_index/prototypes.npy` - Class prototypes
- `faiss_index/labels.npy` - Labels
- `faiss_index/class_names.json` - Class names

---

### Step 3: Test on Exercise 1

```bash
python test_on_exercise1.py
```

**What it does:**
- Loads test images from `exercise_1`
- Computes embeddings
- Matches to nearest prototype
- Applies distance threshold for "unknown" rejection
- Generates confusion matrix and per-class accuracy

**Output:**
- `test_results_exercise1.json` - Evaluation metrics

---

### Step 4: Inference on Single Image

```bash
python inference_faiss.py --image path/to/car.jpg --k 10 --threshold 0.7
```

**Parameters:**
- `--image`: Path to query image
- `--k`: Number of nearest neighbors (default: 10)
- `--threshold`: Unknown rejection threshold (default: 0.7)

**Output:**
- Predicted viewpoint
- Confidence score
- K-NN votes
- Prototype distances

---

## ⚙️ Key Parameters

### Distance Threshold
```python
UNKNOWN_THRESHOLD = 0.7  # In inference_faiss.py and test_on_exercise1.py
```
- Values range from 0 to 1 (cosine similarity)
- Higher = more similar
- If `max(prototype_similarity) < threshold` → predict "unknown"
- Recommended: 0.65 - 0.75

### Embedding Dimension
```python
EMBEDDING_DIM = 256  # In train_embedding_model.py
```
- Options: 128 or 256
- Higher = more expressive, but larger model

### K for K-NN
```python
k = 10  # Number of nearest neighbors
```
- Used for voting and distance aggregation
- Typical range: 5-20

---

## 🧪 Usage Examples

### Training
```bash
# Train embedding model on cfv_viewpoint_train
python train_embedding_model.py

# Build FAISS index
python extract_embeddings_and_build_faiss.py
```

### Testing
```bash
# Evaluate on exercise_1
python test_on_exercise1.py

# Test single image
python inference_faiss.py --image exercise_1/front/car001.jpg
```

### In Production Code
```python
from inference_faiss import predict_viewpoint

result = predict_viewpoint('car.jpg', k=10, verbose=False)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Is known: {result['is_known']}")

if result['is_known']:
    print(f"Canonical viewpoint: {result['predicted_class']}")
else:
    print("Rejected as unknown/invalid viewpoint")
```

---

## 📊 Expected Performance

With good training:
- **Known viewpoints:** 85-95% accuracy
- **Unknown rejection:** Effective with proper threshold tuning
- **Inference speed:** <100ms per image (CPU)

---

## 🔧 Tuning Tips

1. **If too many false unknowns:**
   - Lower threshold (e.g., 0.65)
   - Train with more data augmentation

2. **If accepting bad viewpoints:**
   - Raise threshold (e.g., 0.75)
   - Add more "unknown" class examples

3. **If poor accuracy:**
   - Train longer (increase EPOCHS)
   - Increase embedding dimension (256 → 512)
   - Check data quality

---

## 📦 TFLite Conversion (Optional)

```bash
# Convert embedding model to TFLite
python -c "
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('viewpoint_embedder_savedmodel')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('viewpoint_embedder.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

**Note:** FAISS index stays on server/gateway (CPU inference)

---

## 🎯 Why This Approach?

✅ **Canonical matching** - Direct template matching via prototype vectors  
✅ **Robust rejection** - Distance thresholding for unknown detection  
✅ **Explainable** - Can inspect nearest neighbors and prototype distances  
✅ **Production-ready** - FAISS is battle-tested for billion-scale search  
✅ **No angle regression** - Pure similarity matching  

---

## 📝 File Summary

| File | Purpose |
|------|---------|
| `train_embedding_model.py` | Train MobileNetV2 embedding model |
| `extract_embeddings_and_build_faiss.py` | Build FAISS index from training data |
| `inference_faiss.py` | Single-image inference with FAISS |
| `test_on_exercise1.py` | Evaluate on exercise_1 test set |
| `viewpoint_embedder.h5` | Trained embedding model |
| `faiss_index/` | FAISS index + metadata |

---

**✅ Ready for production deployment!**
