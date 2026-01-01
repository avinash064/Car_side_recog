"""
Car Viewpoint Classifier - Streamlit Demo
Interactive web app for testing the viewpoint classifier
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# Page config
st.set_page_config(
    page_title="Car Viewpoint Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚗 Car Viewpoint Classifier</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a car image to identify its viewpoint using AI</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/TensorFlow-2.10-orange.svg", use_container_width=True)
    st.image("https://img.shields.io/badge/Accuracy-87.22%25-success.svg", use_container_width=True)
    
    st.markdown("## ⚙️ Model Settings")
    
    model_type = st.radio(
        "Select Model Type:",
        ["Float32 (Higher Accuracy)", "INT8 (Faster Inference)"],
        help="Float32: 2.40 MB, higher accuracy\nINT8: 2.59 MB, faster inference"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    if "Float32" in model_type:
        st.info("""
        **Float32 Model**
        - Size: 2.40 MB
        - Accuracy: 87.22%
        - Input: float32 (0-1)
        - Best for: Accuracy
        """)
    else:
        st.info("""
        **INT8 Model**
        - Size: 2.59 MB  
        - Accuracy: ~85-87%
        - Input: uint8 (0-255)
        - Best for: Speed
        """)
    
    st.markdown("---")
    st.markdown("### 📝 Class Labels")
    st.markdown("""
    - 🔵 **Front**
    - 🟢 **Front Left**
    - 🟡 **Front Right**
    - 🔴 **Rear**
    - 🟣 **Rear Left**
    - 🟠 **Rear Right**
    - ⚪ **Unknown**
    """)

# Class names
CLASS_NAMES = ["front", "frontleft", "frontright", "rear", "rearleft", "rearright", "unknown"]
CLASS_EMOJIS = {"front": "🔵", "frontleft": "🟢", "frontright": "🟡", 
                "rear": "🔴", "rearleft": "🟣", "rearright": "🟠", "unknown": "⚪"}

# Load model
@st.cache_resource
def load_model(model_path):
    """Load TFLite model with caching"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Get model path
model_dir = "tflite_models"
if "Float32" in model_type:
    model_path = os.path.join(model_dir, "viewpoint_classifier_float32.tflite")
else:
    model_path = os.path.join(model_dir, "viewpoint_classifier_int8.tflite")

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"❌ Model not found: {model_path}")
    st.info("Please make sure the model files are in the 'tflite_models' directory")
    st.stop()

# Load interpreter
try:
    interpreter = load_model(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success(f"✅ Model loaded: {model_type}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a car image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a car"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image info
        st.info(f"📐 Image size: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.markdown("### 🎯 Prediction Results")
    
    if uploaded_file is not None:
        # Preprocess image
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        
        # Prepare input based on model type
        if "Float32" in model_type:
            # Float32: normalize to 0-1
            input_data = img_array.astype(np.float32) / 255.0
        else:
            # INT8: keep as uint8 (0-255)
            input_data = img_array.astype(np.uint8)
        
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        with st.spinner("🔄 Analyzing image..."):
            start_time = time.time()
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Dequantize if INT8
            if "INT8" in model_type:
                scale = output_details[0]['quantization'][0]
                zero_point = output_details[0]['quantization'][1]
                predictions = (output_data.astype(np.float32) - zero_point) * scale
            else:
                predictions = output_data
            
            inference_time = (time.time() - start_time) * 1000
        
        # Get prediction
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = predictions[predicted_idx]
        
        # Display prediction
        emoji = CLASS_EMOJIS[predicted_class]
        st.markdown(
            f'<div class="prediction-box">{emoji} {predicted_class.upper()}</div>',
            unsafe_allow_html=True
        )
        
        st.metric("Confidence", f"{confidence*100:.2f}%", delta=None)
        st.metric("Inference Time", f"{inference_time:.2f} ms", delta=None)
        
        # Confidence bars
        st.markdown("### 📊 All Class Probabilities")
        for idx, class_name in enumerate(CLASS_NAMES):
            prob = predictions[idx]
            emoji = CLASS_EMOJIS[class_name]
            st.progress(float(prob), text=f"{emoji} {class_name}: {prob*100:.2f}%")
        
    else:
        st.info("👆 Please upload an image to get started")
        
        # Example
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Use clear, well-lit car images
        - Ensure the car is the main subject
        - Avoid heavily cropped or zoomed images
        - Try different angles to see classification
        """)

# Footer
st.markdown("---")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Model Architecture**")
    st.text("MobileNetV2")
    
with col_b:
    st.markdown("**Training Accuracy**")
    st.text("87.22%")
    
with col_c:
    st.markdown("**Parameters**")
    st.text("2.27M")

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">Built with ❤️ using Streamlit and TensorFlow Lite</p>',
    unsafe_allow_html=True
)
