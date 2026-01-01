"""
CFV Dataset to 7-Class Viewpoint Converter
===========================================

This script converts the angle-based CFV dataset to the 7-class viewpoint system:
- front, frontleft, frontright, rear, rearleft, rearright, unknown

Angle mapping logic (0-360 degrees):
- Front: 337.5-22.5° (0° ± 22.5°)
- Front-Right: 22.5-67.5° (45° ± 22.5°)
- Right: 67.5-112.5° (90° ± 22.5°) -> Maps to unknown
- Rear-Right: 112.5-157.5° (135° ± 22.5°)
- Rear: 157.5-202.5° (180° ± 22.5°)
- Rear-Left: 202.5-247.5° (225° ± 22.5°)
- Left: 247.5-292.5° (270° ± 22.5°) -> Maps to unknown
- Front-Left: 292.5-337.5° (315° ± 22.5°)

Author: Senior Computer Vision Engineer
Version: 1.0
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
import shutil


# ============================================================================
# ANGLE TO VIEWPOINT MAPPING
# ============================================================================

def angle_to_viewpoint(angle: float) -> str:
    """
    Convert rotation angle (0-360°) to 7-class viewpoint.
    
    Angle convention: 0° = front view, increases clockwise from bird's eye
    
    Args:
        angle: Rotation angle in degrees (0-360)
        
    Returns:
        Viewpoint class: front, frontleft, frontright, rear, rearleft, rearright, unknown
    """
    # Normalize angle to 0-360 range
    angle = angle % 360
    
    # Define angle ranges for each viewpoint
    # Using 45-degree sectors centered on cardinal/diagonal directions
    if (angle >= 337.5) or (angle < 22.5):
        return 'front'
    elif 22.5 <= angle < 67.5:
        return 'frontright'
    elif 67.5 <= angle < 112.5:
        return 'unknown'  # Pure right side
    elif 112.5 <= angle < 157.5:
        return 'rearright'
    elif 157.5 <= angle < 202.5:
        return 'rear'
    elif 202.5 <= angle < 247.5:
        return 'rearleft'
    elif 247.5 <= angle < 292.5:
        return 'unknown'  # Pure left side
    elif 292.5 <= angle < 337.5:
        return 'frontleft'
    else:
        return 'unknown'


def get_viewpoint_angle_range(viewpoint: str) -> Tuple[float, float]:
    """
    Get the angle range for a given viewpoint.
    
    Args:
        viewpoint: Viewpoint class name
        
    Returns:
        Tuple of (min_angle, max_angle)
    """
    ranges = {
        'front': (337.5, 22.5),  # Wraps around 0
        'frontright': (22.5, 67.5),
        'rearright': (112.5, 157.5),
        'rear': (157.5, 202.5),
        'rearleft': (202.5, 247.5),
        'frontleft': (292.5, 337.5),
        'unknown': None
    }
    return ranges.get(viewpoint)


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

def analyze_cfv_dataset(csv_path: str) -> Dict:
    """
    Analyze CFV dataset and compute statistics.
    
    Args:
        csv_path: Path to train.csv or test.csv
        
    Returns:
        Dictionary with analysis results
    """
    df = pd.read_csv(csv_path)
    
    # Filter valid samples
    valid_df = df[(df['x1'] != 0) | (df['x2'] != 0) | (df['y1'] != 0) | (df['y2'] != 0)].copy()
    
    # Add viewpoint column
    valid_df['viewpoint'] = valid_df['angle'].apply(angle_to_viewpoint)
    
    stats = {
        'total_samples': len(df),
        'valid_samples': len(valid_df),
        'invalid_samples': len(df) - len(valid_df),
        'unique_identities': df['identity'].nunique(),
        'angle_range': (valid_df['angle'].min(), valid_df['angle'].max()),
        'viewpoint_distribution': valid_df['viewpoint'].value_counts().to_dict(),
        'angles_per_identity': valid_df.groupby('identity')['angle'].count().describe().to_dict()
    }
    
    return stats


def print_analysis(stats: Dict, title: str = "CFV Dataset Analysis"):
    """Print formatted analysis results."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"Invalid samples (no bbox): {stats['invalid_samples']}")
    print(f"Unique car identities: {stats['unique_identities']}")
    print(f"Angle range: {stats['angle_range'][0]:.1f}° - {stats['angle_range'][1]:.1f}°")
    print()
    print("Viewpoint Distribution:")
    viewpoint_dist = stats['viewpoint_distribution']
    total_valid = sum(viewpoint_dist.values())
    for viewpoint in ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']:
        count = viewpoint_dist.get(viewpoint, 0)
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        print(f"  {viewpoint:12s}: {count:5d} ({percentage:5.1f}%)")
    print()
    print("Angles per Identity:")
    for key, value in stats['angles_per_identity'].items():
        print(f"  {key}: {value:.1f}")
    print("=" * 80)


# ============================================================================
# CONVERSION PIPELINE
# ============================================================================

def convert_cfv_to_viewpoint_dataset(
    csv_path: str,
    images_dir: str,
    output_dir: str,
    crop_images: bool = True,
    copy_mode: str = 'copy'  # 'copy', 'symlink', or 'move'
) -> Dict:
    """
    Convert CFV dataset to 7-class viewpoint-organized dataset.
    
    Args:
        csv_path: Path to train.csv or test.csv
        images_dir: Path to images directory
        output_dir: Output directory for organized dataset
        crop_images: If True, crop to bounding box; if False, copy full images
        copy_mode: 'copy', 'symlink', or 'move'
        
    Returns:
        Dictionary with conversion statistics
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter valid samples
    valid_df = df[(df['x1'] != 0) | (df['x2'] != 0) | (df['y1'] != 0) | (df['y2'] != 0)].copy()
    
    # Add viewpoint column
    valid_df['viewpoint'] = valid_df['angle'].apply(angle_to_viewpoint)
    
    # Create output directories
    output_path = Path(output_dir)
    viewpoints = ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']
    for vp in viewpoints:
        (output_path / vp).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    stats = defaultdict(int)
    images_base = Path(images_dir)
    
    print(f"Processing {len(valid_df)} valid samples...")
    
    for idx, row in valid_df.iterrows():
        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(valid_df)} images...")
        
        src_path = images_base / row['image_path']
        viewpoint = row['viewpoint']
        
        if not src_path.exists():
            stats['missing_files'] += 1
            continue
        
        # Generate output filename
        # Format: identity_angle_originalname.jpg
        original_name = Path(row['image_path']).name
        identity = int(row['identity'])
        angle = int(row['angle'])
        output_filename = f"{identity:03d}_{angle:03d}_{original_name}"
        output_filepath = output_path / viewpoint / output_filename
        
        try:
            if crop_images:
                # Read and crop image
                image = cv2.imread(str(src_path))
                if image is None:
                    stats['read_failed'] += 1
                    continue
                
                # Extract bbox
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                
                # Validate and clip bbox
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                
                # Crop
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size == 0:
                    stats['empty_crops'] += 1
                    continue
                
                # Save cropped image
                cv2.imwrite(str(output_filepath), cropped)
                stats['cropped'] += 1
            else:
                # Copy/symlink/move full image
                if copy_mode == 'copy':
                    shutil.copy2(src_path, output_filepath)
                elif copy_mode == 'symlink':
                    output_filepath.symlink_to(src_path.absolute())
                elif copy_mode == 'move':
                    shutil.move(str(src_path), str(output_filepath))
                stats['copied'] += 1
            
            stats[viewpoint] += 1
            stats['total_processed'] += 1
            
        except Exception as e:
            print(f"  Error processing {src_path}: {e}")
            stats['errors'] += 1
    
    print(f"  Completed: {stats['total_processed']}/{len(valid_df)}")
    
    return dict(stats)


def print_conversion_stats(stats: Dict):
    """Print formatted conversion statistics."""
    print()
    print("=" * 80)
    print("CONVERSION STATISTICS")
    print("=" * 80)
    print(f"Total processed: {stats.get('total_processed', 0)}")
    print(f"Missing files: {stats.get('missing_files', 0)}")
    print(f"Read failed: {stats.get('read_failed', 0)}")
    print(f"Empty crops: {stats.get('empty_crops', 0)}")
    print(f"Errors: {stats.get('errors', 0)}")
    print()
    print("Images per viewpoint:")
    for vp in ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']:
        count = stats.get(vp, 0)
        print(f"  {vp:12s}: {count:5d}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CFV dataset to 7-class viewpoint system')
    parser.add_argument('--csv', type=str, default='CFV-Dataset/train.csv',
                        help='Path to CSV file (train.csv or test.csv)')
    parser.add_argument('--images', type=str, default='CFV-Dataset/images',
                        help='Path to images directory')
    parser.add_argument('--output', type=str, default='cfv_viewpoint_dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze dataset, do not convert')
    parser.add_argument('--no-crop', action='store_true',
                        help='Do not crop images, copy full images')
    parser.add_argument('--copy-mode', type=str, default='copy', choices=['copy', 'symlink', 'move'],
                        help='How to handle images: copy, symlink, or move')
    
    args = parser.parse_args()
    
    # Analyze dataset
    print("Analyzing CFV dataset...")
    stats = analyze_cfv_dataset(args.csv)
    print_analysis(stats)
    
    if not args.analyze_only:
        # Convert dataset
        print("\nConverting CFV dataset to viewpoint-organized structure...")
        conv_stats = convert_cfv_to_viewpoint_dataset(
            csv_path=args.csv,
            images_dir=args.images,
            output_dir=args.output,
            crop_images=not args.no_crop,
            copy_mode=args.copy_mode
        )
        print_conversion_stats(conv_stats)
        print(f"\nDataset saved to: {args.output}")


if __name__ == "__main__":
    main()
