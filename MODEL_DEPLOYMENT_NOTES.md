# MobileNet FAISS Viewpoint Classifier

## TFLite Models

We provide two TFLite models for deployment:

- **Float32** (2.28 MB) - Higher accuracy
- **INT8** (0.82 MB) - Faster, 64.2% smaller

## Models Not in Repo

**IMPORTANT**: The following files are too large for Git and must be downloaded separately:

### TFLite Models
- `tflite_models/mobilenet_feature_extractor_float32.tflite` (2.28 MB)
- `tflite_models/mobilenet_feature_extractor_int8.tflite` (0.82 MB)

### FAISS Index Files
- `faiss_features/faiss_index.bin` (~40 MB)
- `faiss_features/features.npy` (~38 MB)
- `faiss_features/prototypes.npy` (16 KB)
- `faiss_features/class_names.json` (needed)

## For Streamlit Cloud Deployment

Add these files to `.gitignore` and host them externally (Google Drive, AWS S3, etc.)

Then modify `app_mobilenet.py` to download them on first run.
