# Car Viewpoint Classifier - Training Pipeline

Mobile-ready 6-class car viewpoint classifier using MobileNetV2.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_classifier.py
```

**Outputs:**
- `models/viewpoint_classifier.keras` - Keras model
- `models/saved_model/` - SavedModel format
- `models/best_model.keras` - Best checkpoint
- `models/class_mapping.txt` - Class indices

### 3. Convert to TFLite
```bash
python convert_to_tflite.py
```

**Outputs:**
- `models/tflite/viewpoint_classifier_float32.tflite` - Float32 model
- `models/tflite/viewpoint_classifier_int8.tflite` - INT8 quantized (optional)

### 4. Run Predictions
```bash
python test_predict.py --input path/to/test/images --output predictions.csv
```

**Outputs:**
- CSV file with predictions and confidence scores
- Per-class probability columns

## Model Specifications

- **Architecture:** MobileNetV2 (ImageNet pretrained)
- **Input:** 224×224 RGB
- **Output:** 6 classes (softmax)
- **Classes:** front, frontleft, frontright, rear, rearleft, rearright

## Augmentations

✓ Random brightness (±20%)  
✓ Random translation (±10%)  
✓ Random zoom (±10%)  
❌ **NO horizontal flip** (preserves left/right semantics)

## Training Configuration

- **Batch size:** 32
- **Epochs:** 50 (with early stopping)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical crossentropy
- **Validation split:** 20%

## File Structure

```
car_side_recog/
├── train_classifier.py       # Training script
├── convert_to_tflite.py      # TFLite conversion
├── test_predict.py           # Inference script
├── requirements.txt          # Dependencies
└── models/                   # Output directory
    ├── viewpoint_classifier.keras
    ├── saved_model/
    └── tflite/
        ├── viewpoint_classifier_float32.tflite
        └── viewpoint_classifier_int8.tflite
```

## Notes

- Training data must be organized in class-labeled folders
- No horizontal flip ensures diagonal view integrity
- Fine-tuning last 30 layers for better accuracy
- TFLite models ready for mobile deployment
