# Car Side Recognition - Dataset Visualization Guide

## Overview

I've created a Python visualization tool (`visualize_annotations.py`) that displays car images with their polygon annotations and labels overlaid. This helps you explore and understand the dataset structure before building the classification model.

## Dataset Structure

### Current Status
- **Location**: `c:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog\exercise_1`
- **Total Folders**: 61 subfolders (each representing a different car/session)
- **Total Files**: 4,038 items (images + JSON annotation files)
- **Annotation Format**: VIA VGG Tool JSON format

### Annotation Format Example

Each folder contains:
- **Images**: Car photos from various angles (.jpeg files)
- **Annotations**: `via_region_data.json` file with polygon coordinates and labels

Sample annotation structure:
```json
{
  "image_name.jpeg": {
    "filename": "image_name.jpeg",
    "regions": [
      {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [x1, x2, x3, ...],
          "all_points_y": [y1, y2, y3, ...]
        },
        "region_attributes": {
          "identity": "frontws"  // Label/class name
        }
      }
    ]
  }
}
```

### Annotation Labels Found

Based on the dataset, here are the car part labels annotated:

**Main Parts:**
- `frontws` - Front windshield
- `rearws` / `partial_rearws` - Rear windshield
- `bonnet` / `partial_bonnet` - Hood/bonnet
- `Roof` - Car roof
- `tailgate` / `partial_tailgate` - Rear door/tailgate

**Lighting:**
- `leftheadlamp` / `rightheadlamp` - Headlights
- `lefttaillamp` / `righttaillamp` - Tail lights

**Body Parts:**
- `leftfender` / `rightfender` / `partial_leftfender` - Fenders
- `leftfrontdoor` / `rightfrontdoor` - Front doors
- `leftreardoor` / `rightreardoor` / `partial_leftreardoor` / `partial_rightreardoor` - Rear doors
- `leftqpanel` / `rightqpanel` - Quarter panels

**Bumpers:**
- `frontbumper` / `partial_frontbumper` / `frontbumpergrille` - Front bumper
- `rearbumper` / `partial_rearbumper` - Rear bumper

**Other Components:**
- `wiper` - Windshield wipers
- `logo` - Brand logos
- `namebadge` - Model name badges
- `indicator` - Turn signals
- `leftorvm` / `rightorvm` - Side mirrors (ORVM = Outside Rear View Mirror)
- `doorhandle` - Door handles
- `fuelcap` - Fuel cap
- `sensor` - Parking sensors
- `tyre` / `alloywheel` - Wheels
- `leftrunningboard` / `rightrunningboard` - Running boards
- `leftwa` / `rightwa` - Wheel arches

**Damage/Defects:**
- `scratch` - Scratches
- `dirt` - Dirt marks
- `clipsbroken` - Broken clips
- `d2` - Damage type 2
- `broken` - Generic damage

## Prerequisites

To run the visualization tool, you need:

1. **Python 3.7+**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Required Libraries**
   Install using pip:
   ```bash
   pip install opencv-python numpy
   ```

## How to Use the Visualization Tool

### Basic Usage

Run the script from the command line:

```bash
cd c:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog
python visualize_annotations.py
```

This will:
1. Display statistics about the dataset
2. Visualize the first 3 images from the first subfolder
3. Save annotated images to `visualizations/` folder

### Expected Output

```
Found 61 subfolders in exercise_1
============================================================
Visualizing samples from: 5e9112c35026365e15eb871b
============================================================

Statistics for 5e9112c35026365e15eb871b:
  Total images: 49
  Total annotations: 350+
  Avg annotations per image: 7.14

Annotation types found:
  - frontws: 15
  - wiper: 22
  - Roof: 8
  - bonnet: 5
  ...

============================================================
Creating visualizations...
============================================================

Visualizing IMG_20200219_170506_1586530497634.jpeg
Found 6 annotations
  - frontws: 140 points
  - wiper: 322 points
  - rightapillar: 241 points
  ...
Saved visualization to visualizations/5e9112c35026365e15eb871b/annotated_IMG_...

Visualization complete!
Check the 'visualizations' folder for output images
```

### Customization Options

You can modify the script to:

**1. Visualize specific folders:**
```python
visualizer.visualize_folder("5e9112c35026365e15eb871b", max_images=10)
```

**2. Visualize all folders:**
```python
for subfolder in visualizer.get_all_subfolders():
    visualizer.visualize_folder(subfolder, max_images=2)
```

**3. Get statistics for all folders:**
```python
for subfolder in visualizer.get_all_subfolders():
    stats = visualizer.get_annotation_statistics(subfolder)
    print(f"{subfolder}: {stats}")
```

**4. Visualize specific image:**
```python
vis_img = visualizer.visualize_image(
    "5e9112c35026365e15eb871b",
    "IMG_20200219_170506_1586530497634.jpeg",
    save_output=True
)
```

## Features

### What the Visualizer Does:

1. **Polygon Drawing**: Draws filled, semi-transparent polygons for car parts
2. **Color Coding**: Different colors for different part types
3. **Labels**: Displays part name at the center of each annotation
4. **Statistics**: Shows annotation counts and distribution
5. **Batch Processing**: Can visualize multiple images at once

### Visualization Colors:

- Blue: Front windshield
- Green: Wipers
- Red: Roof
- Cyan: Bonnet
- Magenta: Partial bonnet
- Yellow: Left fender
- Purple: Right fender
- Orange: Left headlamp
- Light Blue: Right headlamp
- White: Logo
- Random colors for other parts

## Understanding the Data for Classification

### For Car Side Recognition Model:

While these annotations show **individual car parts**, your classification task is to identify the **overall car perspective/view**:

1. **Front** - Car viewed from the front
2. **Front Right** - Front-right diagonal view  
3. **Front Left** - Front-left diagonal view
4. **Rear** - Car viewed from the rear
5. **Rear Right** - Rear-right diagonal view
6. **Rear Left** - Rear-left diagonal view

### How to Use These Annotations:

**Option 1: Feature-Based Approach**
- Use the annotated parts as features to determine the view
- E.g., if you see "frontws", "bonnet", "leftheadlamp" → likely "Front Left" view

**Option 2: Ignore Part Annotations**
- Build a standard image classifier using the full images
- Manually label images by view (Front, Rear, etc.)
- Use these labels as ground truth for training

**Option 3: Rule-Based Labeling**
- Analyze which parts appear in each image
- Create rules to automatically assign view labels
- Example rules:
  - "frontws" + "bonnet" + "headlamp" → Front view
  - "rearws" + "tailgate" + "taillamp" → Rear view
  - "leftfender" + "leftorvm" → Left side view

## Next Steps

1. **Install Python** and required libraries (opencv-python, numpy)
2. **Run the visualization** to understand the dataset
3. **Analyze the annotations** to determine how to map them to the 6 view classes
4. **Create a labeling strategy** for car perspectives
5. **Build the classification model** using appropriate architecture (MobileNet, EfficientNet)
6. **Convert to TFLite** for mobile deployment

## Troubleshooting

**Issue**: "Python was not found"
- **Solution**: Install Python from python.org and add to PATH

**Issue**: "ModuleNotFoundError: No module named 'cv2'"
- **Solution**: Run `pip install opencv-python`

**Issue**: "Image not found"
- **Solution**: Verify the exercise_1 folder contains the images

**Issue**: "No annotations found"
- **Solution**: Check that via_region_data.json exists in the subfolder

## Summary

The visualization tool helps you:
- Understand the dataset structure
- See what car parts are annotated
- Explore the quality and consistency of annotations
- Plan your classification approach

You have a rich dataset with detailed polygon annotations for car parts. While the assignment asks for car perspective classification (6 classes), these detailed part annotations can be valuable features or can help you understand the data better.
