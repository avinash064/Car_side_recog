# Car Side Recognition Project

## Overview
This project implements a car orientation/perspective classification system for mobile edge devices using TensorFlow Lite.

## Project Goal
Build a real-time classifier that identifies which of 6 car perspectives an image shows, optimized for mobile deployment.

---

## 🎯 The 6 Car Perspective Classes

### Super-Classes (View-based Categories)
1. **Front** - Front view of the car
2. **Front Right** - Front-right diagonal view
3. **Front Left** - Front-left diagonal view
4. **Rear** - Rear view of the car
5. **Rear Right** - Rear-right diagonal view
6. **Rear Left** - Rear-left diagonal view

### Additional Requirement
- **Random/Negative Class** - Detect and reject images that don't belong to any of the above categories (non-car images, blurry images, or incorrect angles)

---

## 📁 Dataset Structure

### Exercise 1 Dataset
Located in: `exercise_1/`

The dataset contains:
- **4,038 items** including car images and annotations
- Images organized in subfolders (likely by car/session ID)
- **VIA VGG Tool JSON annotations** (`via_region_data.json` files)

#### VIA Annotations Format
The JSON annotations contain polygon coordinates and region attributes that can be used to:
- Extract car parts/regions
- Identify car perspectives
- Filter and organize images by view type

---

## 🔧 Technical Requirements

### Model Requirements
- **Input**: Car images from various perspectives
- **Output**: Classification into one of 6 perspectives + rejection of random images
- **Format**: TensorFlow Lite (.tflite) for mobile deployment
- **Optimization**: Must run efficiently on edge devices

### Performance Metric
- **F1 Score** for each of the 6 image types
- Evaluation on separate test dataset

---

## 📋 Tasks

### 1. Data Preparation
- [ ] Explore the exercise_1 dataset structure
- [ ] Parse VIA JSON annotations
- [ ] Organize images by perspective/view
- [ ] Split data into train/validation/test sets
- [ ] Apply data augmentation techniques

### 2. Model Development
- [ ] Choose appropriate architecture (MobileNet, EfficientNet, etc.)
- [ ] Train classification model
- [ ] Implement negative sample detection
- [ ] Validate model performance
- [ ] Measure F1 scores for each class

### 3. Model Optimization
- [ ] Convert model to TensorFlow Lite
- [ ] Apply quantization (if needed)
- [ ] Test inference speed on target devices

### 4. Inference Pipeline
- [ ] Create `test_pipeline.py` script
- [ ] Implement image loading and preprocessing
- [ ] Integrate TFLite model inference
- [ ] Output predictions with confidence scores

### 5. Documentation
- [ ] Document dataset preparation process
- [ ] Document data augmentation techniques
- [ ] Document model architecture
- [ ] Document training parameters and hyperparameters
- [ ] Create final `readme.txt` for submission

---

## 📦 Deliverables

Final submission: `product_overlay_firstname_lastname.zip`

Must contain:
1. Complete inference pipeline (all code)
2. TFLite model file(s)
3. `readme.txt` with comprehensive documentation

---

## 🚀 Getting Started

### Step 1: Explore Dataset
```bash
# Navigate to dataset directory
cd car_side_recog/exercise_1

# Check dataset structure
# Look for via_region_data.json files
# Count images per category/folder
```

### Step 2: Parse Annotations
- Read VIA JSON files
- Extract image filenames and region data
- Map images to perspective classes

### Step 3: Build Data Pipeline
- Create dataset loader
- Implement augmentation (rotation, flip, color jitter, etc.)
- Prepare TensorFlow/PyTorch data generators

### Step 4: Train Model
- Select backbone architecture
- Configure training parameters
- Monitor validation metrics
- Save best model checkpoints

### Step 5: Convert to TFLite
```python
# Example TFLite conversion
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('model_path')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('car_perspective_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 📊 Suggested Model Architectures

### Option 1: MobileNetV3
- Designed for mobile/edge devices
- Good accuracy-to-size ratio
- Fast inference

### Option 2: EfficientNet-Lite
- Optimized for TFLite
- Multiple size variants
- Excellent performance

### Option 3: Custom CNN
- Lightweight custom architecture
- Tailored to this specific task
- Full control over model size

---

## 🔍 Relationship to Previous Car Projects

### This Project (Car Side Recognition)
- **Goal**: Classify entire car images into 6 perspectives
- **Use Case**: Real-time validation of user-captured photos
- **Output**: Single perspective label per image

### Previous Project (Car Part Cropping)
- **Goal**: Extract and crop specific car parts from images
- **Use Case**: Create refined dataset of car part crops
- **Output**: Cropped images organized by super-class folders
- **Method**: Merged fine-grained part annotations into view-based super-classes

**Key Difference**: 
- Previous project focused on **part segmentation and cropping**
- This project focuses on **whole-image classification**

Both projects use the same 6 super-class views, but serve different purposes!

---

## 📝 Notes

- Ensure proper handling of class imbalance
- Consider edge cases (partially visible cars, occlusions)
- Test on various lighting conditions and backgrounds
- Validate model generalization on unseen data

---

## 📧 Contact
For questions about the assignment, refer to the CV Engineer Assignment @CQ document.
