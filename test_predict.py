"""
Run inference using TFLite model and output predictions as CSV.
Processes a folder of images and generates predictions DataFrame.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from PIL import Image
import argparse

# Configuration
TFLITE_MODEL = r"C:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog\models\tflite\viewpoint_classifier_float32.tflite"
CLASS_NAMES = ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright']
IMG_SIZE = 224


class TFLitePredictor:
    """TFLite inference wrapper."""
    
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Check if model uses quantization
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8
        
        print(f"Model loaded: {os.path.basename(model_path)}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Quantized: {self.is_quantized}")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32)
        
        # MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Quantize if needed
        if self.is_quantized:
            scale, zero_point = self.input_details[0]['quantization']
            img_array = img_array / scale + zero_point
            img_array = img_array.astype(np.uint8)
        
        return img_array
    
    def predict(self, image_path):
        """Run inference on single image."""
        img_array = self.preprocess_image(image_path)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Dequantize if needed
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale
        
        return output
    
    def predict_batch(self, image_paths, show_progress=True):
        """Run inference on multiple images."""
        results = []
        
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            try:
                probabilities = self.predict(img_path)
                pred_idx = np.argmax(probabilities)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probabilities[pred_idx]
                
                results.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    **{f'prob_{cls}': probabilities[i] for i, cls in enumerate(CLASS_NAMES)}
                })
                
                if show_progress and (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{total} images...")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    **{f'prob_{cls}': 0.0 for cls in CLASS_NAMES}
                })
        
        return pd.DataFrame(results)


def collect_images(folder_path, extensions=('.jpg', '.jpeg', '.png')):
    """Collect all images from folder."""
    folder = Path(folder_path)
    images = []
    
    for ext in extensions:
        images.extend(folder.glob(f'*{ext}'))
        images.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description='Run TFLite inference on images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input folder containing images')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--model', type=str, default=TFLITE_MODEL,
                        help='TFLite model path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Car Viewpoint Prediction")
    print("=" * 60)
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Load model
    predictor = TFLitePredictor(args.model)
    
    # Collect images
    print(f"\nScanning images from: {args.input}")
    image_paths = collect_images(args.input)
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Run predictions
    print("\nRunning inference...")
    df = predictor.predict_batch(image_paths)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n✓ Predictions saved to: {args.output}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)
    print(df['predicted_class'].value_counts().to_string())
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    
    # Show sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    print(df[['filename', 'predicted_class', 'confidence']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
