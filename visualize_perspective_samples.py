"""
Visualize sample images from each perspective category
"""

import os
import cv2
from pathlib import Path
import random

def visualize_perspective_samples(organized_dir, samples_per_perspective=3):
    """
    Create a visualization showing sample images from each perspective
    """
    organized_dir = Path(organized_dir)
    output_dir = organized_dir.parent / "perspective_samples"
    output_dir.mkdir(exist_ok=True)
    
    perspectives = ["front", "front_right", "front_left", "rear", "rear_right", "rear_left"]
    
    print(f"{'='*60}")
    print("Creating Perspective Sample Visualizations")
    print(f"{'='*60}\n")
    
    for perspective in perspectives:
        perspective_dir = organized_dir / perspective
        
        if not perspective_dir.exists():
            print(f"Warning: {perspective} folder not found")
            continue
        
        # Get all images in this perspective
        images = list(perspective_dir.glob("*.jpg"))
        
        if not images:
            print(f"No images found in {perspective}")
            continue
        
        # Randomly sample
        sample_images = random.sample(images, min(samples_per_perspective, len(images)))
        
        print(f"{perspective.upper()}: {len(images)} images")
        print(f"  Samples: {len(sample_images)}")
        
        # Create visualization for this perspective
        for idx, img_path in enumerate(sample_images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Add label to image
            h, w = img.shape[:2]
            label = f"{perspective.upper()} - {img_path.name}"
            
            # Draw background rectangle for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(img, (10, 10), (20 + text_size[0], 20 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(img, label, (15, 20 + text_size[1] - 5), font, font_scale, (0, 255, 0), thickness)
            
            # Save annotated image
            output_path = output_dir / f"{perspective}_sample_{idx+1}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"    Saved: {output_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Samples saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    base_dir = r"c:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog"
    organized_dir = Path(base_dir) / "organized_perspectives"
    
    visualize_perspective_samples(organized_dir, samples_per_perspective=4)
