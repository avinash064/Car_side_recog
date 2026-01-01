"""
Visualization Tool for VIA Annotated Car Images
Displays car images with polygon annotations and labels overlaid
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
import random

class VIAAnnotationVisualizer:
    def __init__(self, base_dir):
        """
        Initialize the visualizer with the base directory containing exercise_1 data
        
        Args:
            base_dir: Path to car_side_recog directory
        """
        self.base_dir = Path(base_dir)
        self.exercise_dir = self.base_dir / "exercise_1"
        
        # Color palette for different part labels (BGR format for OpenCV)
        self.colors = {
            'frontws': (255, 0, 0),      # Blue
            'wiper': (0, 255, 0),         # Green
            'Roof': (0, 0, 255),          # Red
            'bonnet': (255, 255, 0),      # Cyan
            'partial_bonnet': (255, 0, 255),  # Magenta
            'leftfender': (0, 255, 255),  # Yellow
            'rightfender': (128, 0, 128), # Purple
            'leftheadlamp': (255, 128, 0), # Orange
            'rightheadlamp': (0, 128, 255), # Light Blue
            'logo': (255, 255, 255),      # White
        }
        
    def get_random_color(self):
        """Generate a random BGR color"""
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def load_annotations(self, subfolder):
        """
        Load VIA annotations from a subfolder
        
        Args:
            subfolder: Name of the subfolder in exercise_1
            
        Returns:
            Dictionary of annotations
        """
        json_path = self.exercise_dir / subfolder / "via_region_data.json"
        
        if not json_path.exists():
            print(f"Warning: No annotation file found at {json_path}")
            return {}
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def visualize_image(self, subfolder, image_name, save_output=True, output_dir=None):
        """
        Visualize a single image with its annotations
        
        Args:
            subfolder: Name of the subfolder containing the image
            image_name: Name of the image file
            save_output: Whether to save the visualization
            output_dir: Directory to save output (default: visualizations/)
            
        Returns:
            Annotated image as numpy array
        """
        # Load annotations
        annotations = self.load_annotations(subfolder)
        
        if image_name not in annotations:
            print(f"Warning: No annotations found for {image_name}")
            return None
        
        # Load image
        image_path = self.exercise_dir / subfolder / image_name
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}")
            return None
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Get annotations for this image
        img_annotations = annotations[image_name]
        regions = img_annotations.get('regions', [])
        
        print(f"\nVisualizing {image_name}")
        print(f"Found {len(regions)} annotations")
        
        # Draw each region
        for idx, region in enumerate(regions):
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # Get the label/identity
            identity = region_attrs.get('identity', f'unknown_{idx}')
            
            # Get polygon points
            all_x = shape_attrs.get('all_points_x', [])
            all_y = shape_attrs.get('all_points_y', [])
            
            if not all_x or not all_y:
                continue
            
            # Create polygon points
            points = np.array([[x, y] for x, y in zip(all_x, all_y)], dtype=np.int32)
            
            # Get color for this identity
            color = self.colors.get(identity, self.get_random_color())
            
            # Draw filled polygon with transparency
            overlay = vis_image.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
            
            # Draw polygon outline
            cv2.polylines(vis_image, [points], isClosed=True, color=color, thickness=2)
            
            # Add label
            # Find a good position for the label (center of bounding box)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            label_pos = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(identity, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, 
                         (label_pos[0] - 5, label_pos[1] - text_height - 5),
                         (label_pos[0] + text_width + 5, label_pos[1] + 5),
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, identity, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"  - {identity}: {len(all_x)} points")
        
        # Save output if requested
        if save_output:
            if output_dir is None:
                output_dir = self.base_dir / "visualizations"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(exist_ok=True)
            
            # Create subfolder in output
            subfolder_output = output_dir / subfolder
            subfolder_output.mkdir(exist_ok=True)
            
            output_path = subfolder_output / f"annotated_{image_name}"
            cv2.imwrite(str(output_path), vis_image)
            print(f"Saved visualization to {output_path}")
        
        return vis_image
    
    def visualize_folder(self, subfolder, max_images=5, save_output=True):
        """
        Visualize multiple images from a folder
        
        Args:
            subfolder: Name of the subfolder to visualize
            max_images: Maximum number of images to visualize
            save_output: Whether to save visualizations
            
        Returns:
            List of annotated images
        """
        annotations = self.load_annotations(subfolder)
        
        if not annotations:
            return []
        
        # Get list of image names
        if max_images is None:
            image_names = list(annotations.keys())
        else:
            image_names = list(annotations.keys())[:max_images]
        
        visualized_images = []
        for image_name in image_names:
            vis_img = self.visualize_image(subfolder, image_name, save_output)
            if vis_img is not None:
                visualized_images.append(vis_img)
        
        return visualized_images
    
    def get_all_subfolders(self):
        """Get list of all subfolders in exercise_1"""
        subfolders = [f.name for f in self.exercise_dir.iterdir() if f.is_dir()]
        return sorted(subfolders)
    
    def get_annotation_statistics(self, subfolder):
        """
        Get statistics about annotations in a subfolder
        
        Args:
            subfolder: Name of the subfolder
            
        Returns:
            Dictionary with statistics
        """
        annotations = self.load_annotations(subfolder)
        
        if not annotations:
            return {}
        
        total_images = len(annotations)
        total_regions = 0
        identity_counts = {}
        
        for img_name, img_data in annotations.items():
            regions = img_data.get('regions', [])
            total_regions += len(regions)
            
            for region in regions:
                identity = region.get('region_attributes', {}).get('identity', 'unknown')
                identity_counts[identity] = identity_counts.get(identity, 0) + 1
        
        return {
            'total_images': total_images,
            'total_regions': total_regions,
            'avg_regions_per_image': total_regions / total_images if total_images > 0 else 0,
            'identity_counts': identity_counts
        }


def main():
    """Main function to demonstrate the visualizer"""
    
    # Initialize visualizer
    base_dir = r"c:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog"
    visualizer = VIAAnnotationVisualizer(base_dir)
    
    # Get all subfolders
    subfolders = visualizer.get_all_subfolders()
    print(f"Found {len(subfolders)} subfolders in exercise_1")
    
    if not subfolders:
        print("No subfolders found!")
        return
    
    # Visualize images from the first subfolder
    print(f"\n{'='*60}")
    print(f"Visualizing samples from: {subfolders[0]}")
    print(f"{'='*60}")
    
    # Get statistics
    stats = visualizer.get_annotation_statistics(subfolders[0])
    print(f"\nStatistics for {subfolders[0]}:")
    print(f"  Total images: {stats.get('total_images', 0)}")
    print(f"  Total annotations: {stats.get('total_regions', 0)}")
    print(f"  Avg annotations per image: {stats.get('avg_regions_per_image', 0):.2f}")
    print(f"\nAnnotation types found:")
    for identity, count in sorted(stats.get('identity_counts', {}).items()):
        print(f"  - {identity}: {count}")
    
    # Visualize ALL images in the folder
    print(f"\n{'='*60}")
    print("Creating visualizations for ALL images...")
    print(f"{'='*60}")
    visualizer.visualize_folder(subfolders[0], max_images=None, save_output=True)
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Check the 'visualizations' folder for output images")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
