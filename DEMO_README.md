# Car Viewpoint Classifier - Streamlit Demo

## 🚀 Quick Start

Run the interactive demo to test the car viewpoint classifier with both Float32 and INT8 models.

### Installation

```bash
# Install streamlit
pip install streamlit

# Or install all requirements
pip install -r requirements.txt
```

### Running the Demo

```bash
# From the project directory
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Features

- **Model Selection**: Choose between Float32 (higher accuracy) and INT8 (faster inference)
- **Image Upload**: Upload car images in JPG, JPEG, or PNG format
- **Real-time Prediction**: Get instant viewpoint classification
- **Confidence Scores**: See probability distribution across all 7 classes
- **Performance Metrics**: View inference time for each prediction

## 🎯 Supported Viewpoints

The classifier can identify 7 different car viewpoints:

1. **Front** - Front view of the car
2. **Front Left** - Front-left diagonal view
3. **Front Right** - Front-right diagonal view  
4. **Rear** - Rear view of the car
5. **Rear Left** - Rear-left diagonal view
6. **Rear Right** - Rear-right diagonal view
7. **Unknown** - Invalid or unrecognized viewpoint

## 📊 Model Comparison

### Float32 Model
- **Size**: 2.40 MB
- **Accuracy**: 87.22%
- **Best for**: Maximum accuracy
- **Input**: float32 (0-1 normalized)

### INT8 Model  
- **Size**: 2.59 MB
- **Accuracy**: ~85-87%
- **Best for**: Faster inference on mobile devices
- **Input**: uint8 (0-255)

## 💡 Usage Tips

For best results:
- Use clear, well-lit car images
- Ensure the car is the main subject
- Avoid heavily cropped or zoomed images
- The car should be clearly visible

## 📂 Required Files

Make sure you have the following files in the `tflite_models/` directory:
- `viewpoint_classifier_float32.tflite`
- `viewpoint_classifier_int8.tflite`

## 🛠️ Troubleshooting

**Model not found error?**
- Ensure model files are in `tflite_models/` folder
- Run the conversion scripts if models are missing

**Slow inference?**
- Try the INT8 model for faster prediction
- Reduce image size before upload

**Import errors?**
- Make sure all requirements are installed: `pip install -r requirements.txt`

## 🌐 Deployment

To deploy the demo online, you can use:
- Streamlit Cloud (free)
- Heroku
- Google Cloud Run
- AWS

## 📧 Support

For issues or questions, please open an issue on GitHub.
