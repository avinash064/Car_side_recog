# Car Viewpoint Classifier 🚗

**High-accuracy car viewpoint classification for mobile deployment**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-87.22%25-success.svg)](#performance)
[![Model Size](https://img.shields.io/badge/Model%20Size-2.40%20MB-blue.svg)](#model-download)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live Demos:**
- 🚀 **[Classification Demo →](https://car-viewpoint-classifier.streamlit.app)** (Trained Model)
- 🔍 **[FAISS Similarity Demo →](https://car-viewpoint-mobilenet.streamlit.app)** (MobileNet Features)

---

## 📥 Model Download

**Pre-trained TFLite Model (Ready for Mobile Deployment)**

- **Download:** [viewpoint_classifier_float32.tflite](https://drive.google.com/file/d/1VSQf0p9JlgmAAPqRoSOKQjh9bmadYSZj/view?usp=sharing) 📦
- **Size:** 2.40 MB
- **Accuracy:** 87.22%
- **Format:** TensorFlow Lite (Float32)
- **Input:** 224×224×3 RGB images
- **Output:** 7 classes (6 viewpoints + unknown)

### 📦 Available Models

| Model | Size | Accuracy | Download Link |
|-------|------|----------|---------------|
| **Float32** | 2.40 MB | 87.22% | [Download](https://drive.google.com/file/d/1VSQf0p9JlgmAAPqRoSOKQjh9bmadYSZj/view?usp=sharing) |
| **INT8** | 2.59 MB | ~85-87% | [Download](https://drive.google.com/file/d/1zptscx9C15SrVkQiFiRc6WaLzgfbvXC0/view?usp=sharing) |

### Class Labels
```python
classes = ["front", "frontleft", "frontright", "rear", "rearleft", "rearright", "unknown"]
```

---

## 🚀 Quick Start

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path="viewpoint_classifier_float32.tflite")
interpreter.allocate_tensors()

# Prepare image
image = Image.open("car.jpg").resize((224, 224))
image_array = np.array(image).astype(np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Run inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], image_array)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Get result
classes = ["front", "frontleft", "frontright", "rear", "rearleft", "rearright", "unknown"]
predicted_class = classes[np.argmax(predictions)]
confidence = np.max(predictions)
print(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **87.22%** |
| **Model Size** | 2.40 MB |
| **Parameters** | 2.27M |
| **Architecture** | MobileNetV2 |

### Per-Class Accuracy
| Viewpoint | Accuracy |
|-----------|----------|
| Front | 96.26% ⭐ |
| Front Right | 92.74% |
| Rear Left | 90.45% |
| Rear Right | 89.50% |
| Rear | 83.04% |
| Unknown | 82.57% |
| Front Left | 81.67% |

---

## 🎯 Project Overview

This project implements a **7-class car viewpoint classifier** optimized for mobile deployment using TensorFlow Lite.

### Viewpoint Classes
1. **Front** - Front view of the car
2. **Front Right** - Front-right diagonal view
3. **Front Left** - Front-left diagonal view
4. **Rear** - Rear view of the car
5. **Rear Right** - Rear-right diagonal view
6. **Rear Left** - Rear-left diagonal view
7. **Unknown** - Invalid/unrecognized viewpoint

---

## 📁 Repository Structure

```
car_side_recog/
├── README.md                          # This file
├── docs/                              # Additional documentation
│   ├── TRAINING_README.md            # Training instructions
│   ├── CLASSIFICATION_RESULTS.md     # Detailed results
│   └── VISUALIZATION_GUIDE.md        # Visualization tools
├── train_viewpoint_classifier_fast.py # Main training script
├── resume_training.py                 # Resume from checkpoint
├── convert_tflite_final.py           # TFLite conversion
├── test_predict.py                    # Inference script
├── requirements.txt                   # Python dependencies
├── class_map.json                     # Class label mapping
└── .gitignore                         # Git ignore rules
```

---

## 📚 Documentation

- **[Training Guide](docs/TRAINING_README.md)** - How to train the model from scratch
- **[Classification Results](docs/CLASSIFICATION_RESULTS.md)** - Detailed performance analysis
- **[Visualization Guide](docs/VISUALIZATION_GUIDE.md)** - Dataset visualization tools

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/avinash064/Car_side_recog.git
cd Car_side_recog

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- TensorFlow 2.10+
- NumPy
- Pillow
- scikit-learn

---

## 💻 Training

To train the model from scratch:

```bash
# Activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run training
python train_viewpoint_classifier_fast.py
```

For detailed training instructions, see [Training Guide](docs/TRAINING_README.md).

---

## 🔄 TFLite Conversion

Convert your trained model to TFLite:

```bash
python convert_tflite_final.py
```

The converted model will be saved to `tflite_models/viewpoint_classifier_float32.tflite`.

---

## 🧪 Testing

Run inference on test images:

```bash
python test_predict.py --image path/to/car_image.jpg
```

---

## 📝 Model Specifications

- **Input Shape:** `[1, 224, 224, 3]`
- **Input Type:** `float32` (normalized 0-1)
- **Output Shape:** `[1, 7]`
- **Output Type:** `float32` (softmax probabilities)
- **Preprocessing:** Resize to 224×224, normalize to [0, 1]

---

## 🎓 Citation

If you use this model or code in your research, please cite:

```bibtex
@misc{car_viewpoint_classifier_2026,
  author = {Avinash Kashyap},
  title = {Car Viewpoint Classifier},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/avinash064/Car_side_recog}
}
```

---

## 📧 Contact

- **Author:** Avinash Kashyap
- **GitHub:** [@avinash064](https://github.com/avinash064)
- **Repository:** [Car_side_recog](https://github.com/avinash064/Car_side_recog)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- MobileNetV2 architecture from TensorFlow
- Training dataset from CFV (Canonical Frontal View) dataset
- TensorFlow Lite for mobile optimization
