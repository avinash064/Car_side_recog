# Canonical Vehicle Viewpoint Classifier - Training Pipeline

Complete production-ready training pipeline for a 7-class vehicle viewpoint classifier using MobileNetV2.

## Overview

This classifier predicts canonical vehicle viewpoints:
- **6 primary views**: `front`, `frontleft`, `frontright`, `rear`, `rearleft`, `rearright`
- **1 rejection class**: `unknown` (invalid/ambiguous viewpoints)

The model learns visual template matching for canonical orientations, not object detection.

---

## Dataset Structure

```
cfv_viewpoint_train/
├── front/           # Front view images
├── frontleft/       # Front-left diagonal view
├── frontright/      # Front-right diagonal view
├── rear/            # Rear view images
├── rearleft/        # Rear-left diagonal view
├── rearright/       # Rear-right diagonal view
└── unknown/         # Invalid/ambiguous viewpoints
```

**Total images**: ~17,500 direction-aligned crops

---

## Model Architecture

| Component | Specification |
|-----------|--------------|
| **Backbone** | MobileNetV2 (ImageNet pretrained) |
| **Input Size** | 224×224×3 RGB |
| **Output** | 7-class softmax |
| **Optimizer** | AdamW |
| **Loss** | Categorical cross-entropy |

---

## Training Strategy

### Two-Stage Training

#### **Stage 1: Warm-up** (4 epochs)
- **Frozen backbone** (only classification head trains)
- Learning rate: `3e-4`
- Purpose: Adapt classification head to viewpoint task

#### **Stage 2: Fine-tuning** (15 epochs)
- **Unfrozen backbone** (full model trains)
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Purpose: Refine feature extraction for viewpoint patterns

### Class Balancing
- Computed class weights prevent `unknown` class from dominating
- Balanced training across all 7 classes

---

## Data Augmentation

### ✅ Allowed Augmentations
- **Brightness jitter**: ±20% (handles lighting variations)
- **Small translation**: ±10% (handles slight misalignments)
- **Small zoom**: ±10% (handles scale variations)

### ❌ Prohibited Augmentations
- **NO horizontal flip** (breaks left/right semantic meaning)
- **NO heavy rotation** (destroys canonical alignment)

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset
Ensure `cfv_viewpoint_train/` exists with 7 subdirectories

---

## Usage

### Train the Model

```bash
python train_viewpoint_classifier.py
```

### Training Process

The script will:
1. **Load dataset** from `cfv_viewpoint_train/`
2. **Compute class weights** for balanced training
3. **Build MobileNetV2 model** with custom classification head
4. **Stage 1**: Train classification head (4 epochs, frozen backbone)
5. **Stage 2**: Fine-tune entire model (15 epochs)
6. **Save outputs**:
   - `viewpoint_model_savedmodel/` — TensorFlow SavedModel
   - `viewpoint_model.h5` — Keras H5 format
   - `class_map.json` — Class index mapping
   - `training_history.png` — Training curves visualization
   - `best_viewpoint_model.h5` — Best checkpoint

### Expected Training Time
- **GPU**: ~15-30 minutes (recommended)
- **CPU**: ~2-4 hours

---

## Output Files

### 1. `viewpoint_model_savedmodel/`
**TensorFlow SavedModel** format for production deployment
```python
import tensorflow as tf
model = tf.keras.models.load_model('viewpoint_model_savedmodel')
```

### 2. `viewpoint_model.h5`
**Keras H5** format for further training or fine-tuning
```python
from tensorflow import keras
model = keras.models.load_model('viewpoint_model.h5')
```

### 3. `class_map.json`
**Class index mapping**
```json
{
  "0": "front",
  "1": "frontleft",
  "2": "frontright",
  "3": "rear",
  "4": "rearleft",
  "5": "rearright",
  "6": "unknown"
}
```

### 4. `training_history.png`
Training and validation curves showing loss and accuracy over epochs

---

## Example Inference

```python
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model('viewpoint_model_savedmodel')

# Load class mapping
with open('class_map.json', 'r') as f:
    class_map = json.load(f)

# Preprocess image
img = Image.open('test_image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Predicted: {class_map[str(class_idx)]}")
print(f"Confidence: {confidence*100:.2f}%")
```

---

## Model Performance

Typical performance on validation set:
- **Accuracy**: 85-95% (depends on dataset quality)
- **Top-2 Accuracy**: 92-98%

Per-class metrics are printed during training for detailed analysis.

---

## Next Steps

### 1. Convert to TFLite (Mobile Deployment)
```python
import tensorflow as tf

# Load SavedModel
model = tf.keras.models.load_model('viewpoint_model_savedmodel')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('viewpoint_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 2. Quantize for Edge Devices (INT8)
```python
def representative_dataset():
    # Use sample images from training set
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()
```

### 3. Deploy to Production
- **Mobile**: Use TFLite with Android ML Kit or iOS Core ML
- **Web**: Use TensorFlow.js
- **Server**: Use TensorFlow Serving or ONNX Runtime

---

## Troubleshooting

### Issue: Low accuracy for specific classes
**Solution**: Check class distribution. If a class has very few images, consider collecting more data or adjusting class weights.

### Issue: Model overfitting
**Solution**: 
- Increase dropout rate in model architecture
- Add more augmentation
- Reduce fine-tuning epochs

### Issue: Training too slow
**Solution**:
- Use GPU acceleration
- Reduce batch size if running out of memory
- Use mixed precision training (`tf.keras.mixed_precision`)

---

## Technical Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Warmup LR | 3e-4 | Higher LR for head-only training |
| Finetune LR | 1e-4 | Lower LR to preserve pretrained features |
| Weight Decay | 1e-4 | L2 regularization for better generalization |
| Batch Size | 32 | Balance between speed and stability |
| Dropout | 0.2 | Prevent overfitting in classification head |

### Callbacks
- **ReduceLROnPlateau**: Automatically reduces LR when validation loss plateaus
- **EarlyStopping**: Stops training if no improvement for 3-5 epochs
- **ModelCheckpoint**: Saves best model based on validation accuracy

---

## License

This training pipeline is provided for educational and research purposes.

---

## Contact

For questions or issues, please refer to project documentation or create an issue in the repository.
