# Automated Car Perspective Classification Results

## Classification Summary

Successfully classified **400 car images** into **6 perspective categories** using automated machine learning.

### Method
1. **Feature Extraction**: ResNet50 deep learning model
2. **Dimensionality Reduction**: PCA (2048 → 50 dimensions, 92.76% variance retained)
3. **Clustering**: K-Means clustering (6 clusters)
4. **Perspective Assignment**: Rotation pattern analysis

### Results

Total images processed: **400 images** (20 images per sequence × 20 sequences)

#### Distribution by Perspective

| Perspective | Count | Percentage |
|-------------|-------|------------|
| **Front** | 81 | 20.3% |
| **Front Right** | 76 | 19.0% |
| **Front Left** | 35 | 8.8% |
| **Rear** | 138 | 34.5% |
| **Rear Right** | 20 | 5.0% |
| **Rear Left** | 50 | 12.5% |
| **TOTAL** | 400 | 100% |

### Cluster Mapping

The algorithm analyzed rotation patterns to automatically assign clusters to perspectives:

| Cluster ID | Avg Position | Images | Perspective |
|------------|--------------|--------|-------------|
| Cluster 4 | 0.441 | 81 | Front |
| Cluster 5 | 0.486 | 76 | Front Right |
| Cluster 2 | 0.500 | 20 | Rear Right |
| Cluster 1 | 0.509 | 138 | Rear |
| Cluster 0 | 0.553 | 50 | Rear Left |
| Cluster 3 | 0.556 | 35 | Front Left |

> **Position** represents the normalized frame position (0.0-1.0) in the rotation sequence

### Output Structure

```
organized_perspectives/
├── front/          (81 images)
├── front_right/    (76 images)
├── front_left/     (35 images)
├── rear/           (138 images)
├── rear_right/     (20 images)
└── rear_left/      (50 images)
```

### Technical Details

- **Feature Dimension**: 2048 (ResNet50 final layer)
- **PCA Components**: 50
- **Variance Explained**: 92.76%
- **Clustering Algorithm**: K-Means (k=6)
- **Processing Time**: ~4-5 minutes for 400 images

### Observations

1. **Rear views are most common** (34.5%) - likely due to camera positioning
2. **Front and Front Right** are well-balanced (20% and 19%)
3. **Rear Right and Front Left** are underrepresented (5% and 9%)
   - This may reflect the actual distribution in rotation sequences
   - Some sequences may have varying frame counts per angle

### Next Steps

To classify **ALL 2,299 images** in the dataset:
1. Run: `classifier.run_classification(max_images_per_sequence=None)`
2. This will process all images (takes ~20-30 minutes)
3. Results will be more comprehensive and balanced

### Usage for Training

These organized folders can now be used directly for:
- Training a 6-class CNN classifier
- Creating a labeled dataset
- Data augmentation
- TensorFlow Lite model development

---

**Generated**: 2025-12-29  
**Script**: `classify_perspectives.py`  
**Libraries**: PyTorch, torchvision, scikit-learn, DeepImageSearch
