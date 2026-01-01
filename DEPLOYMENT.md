# Deploying to Streamlit Cloud

## 🚀 Deploy Your Own Instance

### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Fork or Push to GitHub** ✅ (Already done!)
   - Repository: https://github.com/avinash064/Car_side_recog

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App**
   - Click "New app"
   - Repository: `avinash064/Car_side_recog`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Download Models**
   - Since models are too large for Git, you'll need to:
   - Download models from Google Drive (links in README)
   - Upload to a cloud storage (Google Drive, AWS S3, etc.)
   - Update `app.py` to download models on startup

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/avinash064/Car_side_recog.git
cd Car_side_recog

# Download models
# Download from Google Drive links in README
# Place in tflite_models/ folder

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 📝 Fix for Protobuf Issue

If you encounter protobuf errors, run:

```bash
pip install protobuf==3.20.3
```

This ensures compatibility between TensorFlow 2.10 and Streamlit.

## 🔧 For Streamlit Cloud Deployment

Since model files are too large for GitHub, you have two options:

### Option A: Use Git LFS
```bash
git lfs install
git lfs track "*.tflite"
git add .gitattributes
git add tflite_models/*.tflite
git commit -m "Add models with Git LFS"
git push
```

### Option B: Download on Startup (Recommended)
Update `app.py` to download models from Google Drive when the app starts:

```python
import gdown

# Add this before loading models
if not os.path.exists("tflite_models/viewpoint_classifier_float32.tflite"):
    gdown.download(
        "https://drive.google.com/uc?id=1VSQf0p9JlgmAAPqRoSOKQjh9bmadYSZj",
        "tflite_models/viewpoint_classifier_float32.tflite"
    )

if not os.path.exists("tflite_models/viewpoint_classifier_int8.tflite"):
    gdown.download(
        "https://drive.google.com/uc?id=1zptscx9C15SrVkQiFiRc6WaLzgfbvXC0",
        "tflite_models/viewpoint_classifier_int8.tflite"
    )
```

Then add `gdown` to requirements.txt.

## 🌐 Your Deployed App

Once deployed, your app will be available at:
`https://[your-custom-name].streamlit.app`

## 📧 Need Help?

Check the [Streamlit documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started)
