"""
Car Viewpoint Classifier Streamlit App
Uses MobileNet features + FAISS for canonical viewpoint matching
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import time

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page config
st.set_page_config(
    page_title="Car Viewpoint Classifier",
    page_icon="Car",
    layout="wide"
)

# Configuration
FAISS_DIR = "faiss_features"
TFLITE_DIR = "tflite_models"
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.75

# ============================================================================
# Load Resources
# ============================================================================

@st.cache_resource
def load_resources(model_type="float32"):
    """Load FAISS index, prototypes, and TFLite model"""
    
    # For demo without faiss-cpu on streamlit cloud
    try:
        import faiss as faiss_module
        HAS_FAISS = True
    except:
        HAS_FAISS = False
        faiss_module = None
    
    # Load class names
    with open(os.path.join(FAISS_DIR, 'class_names.json'), 'r') as f:
        class_names = json.load(f)
    
    # Load prototypes
    prototypes = np.load(os.path.join(FAISS_DIR, 'prototypes.npy'))
    
    # Load TFLite model
    if model_type == "float32":
        model_path = os.path.join(TFLITE_DIR, "mobilenet_feature_extractor_float32.tflite")
    else:
        model_path = os.path.join(TFLITE_DIR, "mobilenet_feature_extractor_int8.tflite")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return {
        'class_names': class_names,
        'prototypes': prototypes,
        'interpreter': interpreter,
        'model_type': model_type,
        'has_faiss': HAS_FAISS
    }

# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features_tflite(image, interpreter, model_type):
    """Extract features using TFLite model"""
    
    # Resize image
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input based on model type
    if model_type == "float32":
        # Float32: MobileNet preprocessing
        input_data = img_array.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = (input_data - 127.5) / 127.5  # Normalize to [-1, 1]
    else:
        # INT8: keep as uint8
        input_data = img_array.astype(np.uint8)
        input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    features = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if INT8
    if model_type == "int8":
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        features = (features.astype(np.float32) - zero_point) * scale
    
    # L2 normalize
    features = features / np.linalg.norm(features)
    
    return features

def predict_viewpoint(image, resources):
    """Predict viewpoint using prototype matching"""
    
    # Extract features
    features = extract_features_tflite(
        image, 
        resources['interpreter'],
        resources['model_type']
    )
    
    # Compute similarity to prototypes
    similarities = np.dot(features, resources['prototypes'].T)[0]
    
    # Best match
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    best_class = resources['class_names'][best_idx]
    
    # Apply threshold
    if best_similarity < UNKNOWN_THRESHOLD:
        predicted_class = 'unknown'
        confidence = best_similarity
    else:
        predicted_class = best_class
        confidence = best_similarity
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'best_match': best_class,
        'best_similarity': best_similarity,
        'all_similarities': {cn: float(similarities[i]) 
                            for i, cn in enumerate(resources['class_names'])}
    }

# ============================================================================
# UI
# ============================================================================

# Title
st.title("Car Viewpoint Classifier")
st.markdown("### MobileNet + FAISS Similarity Matching")

# Sidebar
with st.sidebar:
    st.markdown("## Settings")
    
    model_type = st.radio(
        "Model Type:",
        ["float32", "int8"],
        format_func=lambda x: f"Float32 (2.28 MB)" if x == "float32" else "INT8 (0.82 MB)"
    )
    
    st.markdown("---")
    st.markdown(f"**Threshold:** {UNKNOWN_THRESHOLD}")
    st.markdown(f"**Feature Dim:** 576")
    
    st.markdown("---")
    st.markdown("### Classes")
    st.markdown("""
    - Front
    - Front Left
    - Front Right
    - Rear
    - Rear Left
    - Rear Right
    - Unknown
    """)

# Load resources
resources = load_resources(model_type)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a car image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### Prediction")
    
    if uploaded_file:
        with st.spinner("Analyzing..."):
            start_time = time.time()
            result = predict_viewpoint(image, resources)
            inference_time = (time.time() - start_time) * 1000
        
        # Display result
        st.markdown(f"## **{result['predicted_class'].upper()}**")
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")
        st.metric("Inference Time", f"{inference_time:.0f} ms")
        
        # Similarity scores
        st.markdown("### Prototype Similarities")
        for class_name in sorted(result['all_similarities'].keys(), 
                                 key=lambda x: result['all_similarities'][x], 
                                 reverse=True):
            sim = result['all_similarities'][class_name]
            st.progress(float(sim), text=f"{class_name}: {sim*100:.1f}%")
    
    else:
        st.info("Upload an image to get started")

# Footer
st.markdown("---")
st.markdown("""
**System Info:**
- Model: MobileNetV2 (pre-trained ImageNet)
- Matching: FAISS Cosine Similarity  
- Features: 576-dim L2-normalized vectors
- Threshold: Distance-based unknown rejection
""")
