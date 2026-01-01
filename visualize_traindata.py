"""
Visualization Tool for EPFL GIMS-08 Bounding Box Annotations
Displays car images with bounding box annotations overlaid
"""

import os
import cv2
import numpy as np
from pathlib import Path

class EPFLBBoxVisualizer:
    def __init__(self, base_dir):
        """
        Initialize the visualizer with the base directory containing traindata
        
        Args:
            base_dir: Path to car_side_recog directory
        """
        self.base_dir = Path(base_dir)
        self.traindata_dir = self.base_dir / "traindata" / "epfl-gims08" / "tripod-seq"
        
    def load_bboxes(self, sequence_num):
        """
        Load bounding boxes for a specific sequence
        
        Args:
            sequence_num: Sequence number (1-20)
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        bbox_file = self.traindata_dir / f"bbox_{sequence_num:02d}.txt"
        
        if not bbox_file.exists():
            print(f"Warning: No bbox file found at {bbox_file}")
            return []
        
        bboxes = []
        with open(bbox_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    coords = list(map(float, line.split()))
                    if len(coords) == 4:
                        bboxes.append(coords)
                except ValueError:
                    continue
        
        return bboxes
    
    def visualize_sequence(self, sequence_num, max_images=10, save_output=True, output_dir=None):
        """
        Visualize images from a sequence with their bounding boxes
        
        Args:
            sequence_num: Sequence number (1-20)
            max_images: Maximum number of images to visualize  
            save_output: Whether to save visualizations
            output_dir: Directory to save output (default: traindata_visualizations/)
            
        Returns:
            List of annotated images
        """
        # Load bounding boxes
        bboxes = self.load_bboxes(sequence_num)
        
        if not bboxes:
            print(f"No bounding boxes found for sequence {sequence_num}")
            return []
        
        print(f"\nVisualizing sequence {sequence_num:02d}")
        print(f"Found {len(bboxes)} bounding boxes")
        
        # Setup output directory
        if save_output:
            if output_dir is None:
                output_dir = self.base_dir / "traindata_visualizations"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(exist_ok=True)
            seq_output = output_dir / f"tripod_seq_{sequence_num:02d}"
            seq_output.mkdir(exist_ok=True)
        
        visualized_images = []
        num_to_process = min(len(bboxes), max_images) if max_images else len(bboxes)
        
        for i in range(num_to_process):
            # Get image path
            image_name = f"tripod_seq_{sequence_num:02d}_{i+1:03d}.jpg"
            image_path = self.traindata_dir / image_name
            
            if not image_path.exists():
                print(f"Warning: Image not found at {image_path}")
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            # Create visualization
            vis_image = image.copy()
            
            # Get bounding box for this image
            if i < len(bboxes):
                x1, y1, x2, y2 = bboxes[i]
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Car (seq {sequence_num:02d})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(vis_image, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0] + 5, y1),
                            (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                print(f"  Image {i+1:03d}: BBox [{x1}, {y1}, {x2}, {y2}]")
            
            visualized_images.append(vis_image)
            
            # Save output if requested
            if save_output:
                output_path = seq_output / f"annotated_{image_name}"
                cv2.imwrite(str(output_path), vis_image)
        
        if save_output:
            print(f"Saved {len(visualized_images)} visualizations to {seq_output}")
        
        return visualized_images
    
    def get_statistics(self):
        """
        Get statistics about all sequences
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        total_images = 0
        total_boxes = 0
        
        for seq_num in range(1, 21):
            bboxes = self.load_bboxes(seq_num)
            if bboxes:
                stats[f"seq_{seq_num:02d}"] = len(bboxes)
                total_boxes += len(bboxes)
                total_images += len(bboxes)  # One bbox per image
        
        stats['total_sequences'] = len([k for k in stats.keys() if k.startswith('seq_')])
        stats['total_images'] = total_images
        stats['total_boxes'] = total_boxes
        
        return stats
    
    def visualize_all_sequences(self, max_per_sequence=5, save_output=True):
        """
        Visualize samples from all sequences
        
        Args:
            max_per_sequence: Maximum images per sequence
            save_output: Whether to save visualizations
        """
        print(f"{'='*60}")
        print("Visualizing all sequences")
        print(f"{'='*60}")
        
        for seq_num in range(1, 21):
            try:
                print(f"\n--- Sequence {seq_num:02d} ---")
                self.visualize_sequence(seq_num, max_images=max_per_sequence, save_output=save_output)
            except Exception as e:
                print(f"Error processing sequence {seq_num}: {e}")
                continue


def main():
    """Main function to demonstrate the visualizer"""
    
    # Initialize visualizer
    base_dir = r"c:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog"
    visualizer = EPFLBBoxVisualizer(base_dir)
    
    # Get statistics
    print(f"{'='*60}")
    print("EPFL GIMS-08 Dataset Statistics")
    print(f"{'='*60}")
    stats = visualizer.get_statistics()
    
    print(f"\nTotal sequences: {stats['total_sequences']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Total bounding boxes: {stats['total_boxes']}")
    
    print(f"\nImages per sequence:")
    for key, value in sorted(stats.items()):
        if key.startswith('seq_'):
            seq_num = key.split('_')[1]
            print(f"  Sequence {seq_num}: {value} images")
    
    # Visualize ALL sequences with first 3 images each
    print(f"\n{'='*60}")
    print("Creating visualizations for ALL 20 sequences...")
    print(f"{'='*60}")
    
    for seq_num in range(1, 21):
        print(f"\nProcessing sequence {seq_num:02d}...")
        visualizer.visualize_sequence(seq_num, max_images=3, save_output=True)
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Check the 'traindata_visualizations' folder for output images")
    print(f"{'='*60}")
    
    print(f"\nThis dataset contains car images captured from a tripod at different angles.")
    print(f"Each sequence shows a car from a continuously rotating viewpoint.")
    print(f"Bounding boxes mark the car location in each frame.")


if __name__ == "__main__":
    main()
