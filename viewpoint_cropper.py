"""
Car Viewpoint Inference and Cropping System
============================================

This script processes a dataset of car images with VIA polygon annotations to:
1. Infer viewpoint super-class labels from part annotations
2. Crop direction-aware ROIs
3. Save cropped images into a structured output dataset

Author: Senior Computer Vision Engineer
Version: 1.0
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import cv2
import numpy as np
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

# Noise classes to completely ignore
NOISE_CLASSES = {
    'scratch', 'sensor', 'wheel', 'wheelcap', 'tyre',
    'roof', 'logo', 'indicator', 'reflector', 'dirt'
}

# Part-to-directional evidence mapping
FRONT_PARTS = {'bonnet', 'partial_bonnet', 'frontws', 'wiper'}
REAR_PARTS = {'rearws', 'tailgate', 'rearbumper', 'rearbumpercladding', 'taillamp', 'licenseplate'}
LEFT_PARTS = {'leftorvm', 'leftdoor', 'leftapillar', 'leftroofside', 'leftcpillar'}
RIGHT_PARTS = {'rightorvm', 'rightdoor', 'rightapillar', 'rightroofside', 'rightcpillar'}

# Super-class labels
SUPER_CLASSES = ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']

# Cropping ratios (percentage of ROI to keep)
CROP_RATIO = 0.6


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_identity(identity: str) -> str:
    """Normalize identity string: lowercase and strip whitespace."""
    return identity.lower().strip()


def extract_parts_from_via(via_data: Dict) -> Dict[str, Set[str]]:
    """
    Extract normalized part identities from VIA annotation data.
    
    Args:
        via_data: Loaded VIA JSON data
        
    Returns:
        Dictionary mapping image filenames to sets of normalized part identities
    """
    image_parts = defaultdict(set)
    
    # VIA format: via_data is a dict with image keys
    for img_key, img_data in via_data.items():
        if not isinstance(img_data, dict):
            continue
            
        filename = img_data.get('filename')
        regions = img_data.get('regions', [])
        
        if not filename:
            continue
        
        for region in regions:
            region_attrs = region.get('region_attributes', {})
            identity = region_attrs.get('identity', '')
            
            if identity:
                normalized = normalize_identity(identity)
                # Skip noise classes
                if normalized not in NOISE_CLASSES:
                    image_parts[filename].add(normalized)
    
    return dict(image_parts)


def infer_viewpoint(parts: Set[str]) -> str:
    """
    Infer viewpoint super-class from part identities using deterministic rules.
    
    Args:
        parts: Set of normalized part identities
        
    Returns:
        One of the 7 super-class labels
    """
    # Check directional evidence
    has_front = bool(parts & FRONT_PARTS)
    has_rear = bool(parts & REAR_PARTS)
    has_left = bool(parts & LEFT_PARTS)
    has_right = bool(parts & RIGHT_PARTS)
    
    # Apply deterministic rules
    if has_front and has_left and not has_rear and not has_right:
        return 'frontleft'
    elif has_front and has_right and not has_rear and not has_left:
        return 'frontright'
    elif has_rear and has_left and not has_front and not has_right:
        return 'rearleft'
    elif has_rear and has_right and not has_front and not has_left:
        return 'rearright'
    elif has_front and not has_rear and not has_left and not has_right:
        return 'front'
    elif has_rear and not has_front and not has_left and not has_right:
        return 'rear'
    else:
        # Ambiguous or weak evidence
        return 'unknown'


def get_polygon_bbox(polygon_x: List[int], polygon_y: List[int]) -> Tuple[int, int, int, int]:
    """
    Get bounding box from polygon coordinates.
    
    Args:
        polygon_x: List of x coordinates
        polygon_y: List of y coordinates
        
    Returns:
        Tuple of (x_min, y_min, x_max, y_max)
    """
    x_min = int(min(polygon_x))
    y_min = int(min(polygon_y))
    x_max = int(max(polygon_x))
    y_max = int(max(polygon_y))
    return x_min, y_min, x_max, y_max


def merge_all_polygons(via_data: Dict, filename: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Merge all polygon annotations for an image to compute base car ROI.
    
    Args:
        via_data: Loaded VIA JSON data
        filename: Image filename to process
        
    Returns:
        Merged bounding box (x_min, y_min, x_max, y_max) or None if no valid polygons
    """
    all_x = []
    all_y = []
    
    for img_key, img_data in via_data.items():
        if not isinstance(img_data, dict):
            continue
            
        if img_data.get('filename') != filename:
            continue
        
        regions = img_data.get('regions', [])
        
        for region in regions:
            region_attrs = region.get('region_attributes', {})
            identity = region_attrs.get('identity', '')
            
            if identity:
                normalized = normalize_identity(identity)
                # Skip noise classes
                if normalized in NOISE_CLASSES:
                    continue
            
            shape_attrs = region.get('shape_attributes', {})
            if shape_attrs.get('name') == 'polygon':
                all_x.extend(shape_attrs.get('all_points_x', []))
                all_y.extend(shape_attrs.get('all_points_y', []))
    
    if not all_x or not all_y:
        return None
    
    return get_polygon_bbox(all_x, all_y)


def apply_direction_crop(roi_bbox: Tuple[int, int, int, int], viewpoint: str) -> Tuple[int, int, int, int]:
    """
    Apply direction-aware cropping to ROI based on viewpoint.
    
    Args:
        roi_bbox: Base ROI (x_min, y_min, x_max, y_max)
        viewpoint: Inferred viewpoint label
        
    Returns:
        Cropped bounding box (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = roi_bbox
    width = x_max - x_min
    height = y_max - y_min
    
    if viewpoint == 'front':
        # Top 60% of ROI
        new_height = int(height * CROP_RATIO)
        return x_min, y_min, x_max, y_min + new_height
    
    elif viewpoint == 'rear':
        # Bottom 60% of ROI
        new_height = int(height * CROP_RATIO)
        return x_min, y_max - new_height, x_max, y_max
    
    elif viewpoint == 'frontleft':
        # Top-left 60% of ROI
        new_width = int(width * CROP_RATIO)
        new_height = int(height * CROP_RATIO)
        return x_min, y_min, x_min + new_width, y_min + new_height
    
    elif viewpoint == 'frontright':
        # Top-right 60% of ROI
        new_width = int(width * CROP_RATIO)
        new_height = int(height * CROP_RATIO)
        return x_max - new_width, y_min, x_max, y_min + new_height
    
    elif viewpoint == 'rearleft':
        # Bottom-left 60% of ROI
        new_width = int(width * CROP_RATIO)
        new_height = int(height * CROP_RATIO)
        return x_min, y_max - new_height, x_min + new_width, y_max
    
    elif viewpoint == 'rearright':
        # Bottom-right 60% of ROI
        new_width = int(width * CROP_RATIO)
        new_height = int(height * CROP_RATIO)
        return x_max - new_width, y_max - new_height, x_max, y_max
    
    else:  # 'unknown'
        # Full ROI
        return roi_bbox


def crop_and_save_image(
    image_path: Path,
    crop_bbox: Tuple[int, int, int, int],
    output_dir: Path,
    viewpoint: str
) -> bool:
    """
    Crop image and save to appropriate viewpoint folder.
    
    Args:
        image_path: Path to source image
        crop_bbox: Crop bounding box (x_min, y_min, x_max, y_max)
        output_dir: Root output directory
        viewpoint: Viewpoint label (determines subfolder)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [!] Failed to read image: {image_path}")
            return False
        
        # Validate crop bounds
        x_min, y_min, x_max, y_max = crop_bbox
        img_height, img_width = image.shape[:2]
        
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(x_min + 1, min(x_max, img_width))
        y_max = max(y_min + 1, min(y_max, img_height))
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            print(f"  [!] Empty crop for image: {image_path}")
            return False
        
        # Create output directory
        viewpoint_dir = output_dir / viewpoint
        viewpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename (preserve original name, prepend subfolder if needed)
        output_filename = image_path.name
        output_path = viewpoint_dir / output_filename
        
        # Handle filename conflicts
        counter = 1
        while output_path.exists():
            stem = image_path.stem
            suffix = image_path.suffix
            output_filename = f"{stem}_{counter}{suffix}"
            output_path = viewpoint_dir / output_filename
            counter += 1
        
        # Save cropped image
        cv2.imwrite(str(output_path), cropped)
        return True
        
    except Exception as e:
        print(f"  [X] Error processing {image_path}: {e}")
        return False


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_subfolder(subfolder_path: Path, output_dir: Path) -> Dict[str, int]:
    """
    Process a single subfolder containing images and VIA annotations.
    
    Args:
        subfolder_path: Path to subfolder
        output_dir: Root output directory
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {label: 0 for label in SUPER_CLASSES}
    stats['total_processed'] = 0
    stats['total_failed'] = 0
    
    # Look for VIA annotation file
    via_file = subfolder_path / 'via_region_data.json'
    if not via_file.exists():
        print(f"  [!] No via_region_data.json found in {subfolder_path.name}")
        return stats
    
    # Load VIA annotations
    try:
        with open(via_file, 'r', encoding='utf-8') as f:
            via_data = json.load(f)
    except Exception as e:
        print(f"  [X] Failed to load {via_file}: {e}")
        return stats
    
    # Extract parts for each image
    image_parts = extract_parts_from_via(via_data)
    
    if not image_parts:
        print(f"  [!] No valid annotations found in {subfolder_path.name}")
        return stats
    
    # Process each annotated image
    for filename, parts in image_parts.items():
        image_path = subfolder_path / filename
        
        if not image_path.exists():
            print(f"  [!] Image not found: {filename}")
            stats['total_failed'] += 1
            continue
        
        # Infer viewpoint
        viewpoint = infer_viewpoint(parts)
        
        # Get merged ROI
        roi_bbox = merge_all_polygons(via_data, filename)
        if roi_bbox is None:
            print(f"  [!] No valid polygons for {filename}")
            stats['total_failed'] += 1
            continue
        
        # Apply direction-aware cropping
        crop_bbox = apply_direction_crop(roi_bbox, viewpoint)
        
        # Crop and save
        success = crop_and_save_image(image_path, crop_bbox, output_dir, viewpoint)
        
        if success:
            stats[viewpoint] += 1
            stats['total_processed'] += 1
        else:
            stats['total_failed'] += 1
    
    return stats


def process_dataset(root_folder: str, output_folder: str) -> None:
    """
    Main entry point: process entire dataset folder-by-folder.
    
    Args:
        root_folder: Path to root dataset folder
        output_folder: Path to output cropped dataset folder
    """
    root_path = Path(root_folder)
    output_path = Path(output_folder)
    
    if not root_path.exists():
        print(f"[X] Root folder does not exist: {root_folder}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CAR VIEWPOINT INFERENCE AND CROPPING SYSTEM")
    print("=" * 80)
    print(f"Root folder: {root_path.absolute()}")
    print(f"Output folder: {output_path.absolute()}")
    print()
    
    # Find all subfolders containing via_region_data.json
    subfolders = []
    for item in root_path.iterdir():
        if item.is_dir():
            via_file = item / 'via_region_data.json'
            if via_file.exists():
                subfolders.append(item)
    
    if not subfolders:
        print("[X] No subfolders with via_region_data.json found!")
        return
    
    print(f"Found {len(subfolders)} subfolders to process\n")
    
    # Process each subfolder
    global_stats = {label: 0 for label in SUPER_CLASSES}
    global_stats['total_processed'] = 0
    global_stats['total_failed'] = 0
    
    for idx, subfolder in enumerate(subfolders, 1):
        print(f"[{idx}/{len(subfolders)}] Processing: {subfolder.name}")
        
        stats = process_subfolder(subfolder, output_path)
        
        # Update global statistics
        for key, value in stats.items():
            global_stats[key] += value
        
        # Print subfolder summary
        processed = stats['total_processed']
        failed = stats['total_failed']
        print(f"  [+] Processed: {processed} | Failed: {failed}")
        
        if processed > 0:
            viewpoint_summary = ", ".join([f"{label}: {stats[label]}" for label in SUPER_CLASSES if stats[label] > 0])
            print(f"  -> {viewpoint_summary}")
        
        print()
    
    # Print final summary
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total images processed: {global_stats['total_processed']}")
    print(f"Total images failed: {global_stats['total_failed']}")
    print()
    print("Viewpoint distribution:")
    for label in SUPER_CLASSES:
        count = global_stats[label]
        if count > 0:
            percentage = (count / global_stats['total_processed'] * 100) if global_stats['total_processed'] > 0 else 0
            print(f"  {label:12s}: {count:5d} ({percentage:5.1f}%)")
    print("=" * 80)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Default paths (modify as needed)
    DEFAULT_ROOT = "ROOT_FOLDER"  # Change this to your dataset root
    DEFAULT_OUTPUT = "cropped_dataset"
    
    # Parse command-line arguments
    if len(sys.argv) >= 2:
        root_folder = sys.argv[1]
    else:
        root_folder = DEFAULT_ROOT
    
    if len(sys.argv) >= 3:
        output_folder = sys.argv[2]
    else:
        output_folder = DEFAULT_OUTPUT
    
    # Run processing pipeline
    process_dataset(root_folder, output_folder)
