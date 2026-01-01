"""
CFV Viewpoint Visualization Tool
=================================

Generates visual reports to verify angle-to-viewpoint mapping quality.

Features:
- Sample grid for each viewpoint class
- Angle distribution histograms with sector boundaries
- HTML report with embedded images

Author: Senior Computer Vision Engineer
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64
from io import BytesIO


def load_sample_images_per_viewpoint(csv_path: str, images_dir: str, samples_per_class: int = 16) -> dict:
    """Load sample images for each viewpoint class."""
    df = pd.read_csv(csv_path)
    
    # Filter valid samples
    valid_df = df[(df['x1'] != 0) | (df['x2'] != 0) | (df['y1'] != 0) | (df['y2'] != 0)].copy()
    
    # Import angle mapping function
    from cfv_to_viewpoint import angle_to_viewpoint
    
    valid_df['viewpoint'] = valid_df['angle'].apply(angle_to_viewpoint)
    
    images_base = Path(images_dir)
    samples = {}
    
    viewpoints = ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']
    
    for vp in viewpoints:
        vp_df = valid_df[valid_df['viewpoint'] == vp].sample(n=min(samples_per_class, len(valid_df[valid_df['viewpoint'] == vp])))
        
        vp_samples = []
        for _, row in vp_df.iterrows():
            img_path = images_base / row['image_path']
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Crop to bbox
                    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size > 0:
                        # Convert BGR to RGB
                        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        vp_samples.append({
                            'image': cropped_rgb,
                            'angle': row['angle'],
                            'identity': row['identity']
                        })
        
        samples[vp] = vp_samples
    
    return samples


def create_viewpoint_grid(samples: dict, cols: int = 4) -> plt.Figure:
    """Create a grid showing samples for each viewpoint."""
    viewpoints = ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']
    
    rows_per_vp = int(np.ceil(16 / cols))
    total_rows = len(viewpoints) * (rows_per_vp + 1)  # +1 for title row
    
    fig = plt.figure(figsize=(cols * 3, total_rows * 2.5))
    
    for vp_idx, vp in enumerate(viewpoints):
        vp_samples = samples.get(vp, [])
        
        # Add viewpoint title
        title_row = vp_idx * (rows_per_vp + 1)
        ax_title = plt.subplot2grid((total_rows, cols), (title_row, 0), colspan=cols, fig=fig)
        ax_title.axis('off')
        count = len(vp_samples)
        ax_title.text(0.5, 0.5, f'{vp.upper()} ({count} samples)', 
                     ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Add sample images
        for img_idx, sample in enumerate(vp_samples[:16]):
            row = title_row + 1 + (img_idx // cols)
            col = img_idx % cols
            
            ax = plt.subplot2grid((total_rows, cols), (row, col), fig=fig)
            ax.imshow(sample['image'])
            ax.set_title(f"Angle: {sample['angle']:.0f}°\nID: {sample['identity']}", fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_angle_distribution_plot(csv_path: str) -> plt.Figure:
    """Create angle distribution histogram with viewpoint sector boundaries."""
    df = pd.read_csv(csv_path)
    valid_df = df[(df['x1'] != 0) | (df['x2'] != 0) | (df['y1'] != 0) | (df['y2'] != 0)].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot histogram
    ax.hist(valid_df['angle'], bins=72, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add sector boundaries
    sectors = [
        (337.5, 22.5, 'front', 'green'),
        (22.5, 67.5, 'frontright', 'orange'),
        (67.5, 112.5, 'unknown (right)', 'gray'),
        (112.5, 157.5, 'rearright', 'red'),
        (157.5, 202.5, 'rear', 'darkred'),
        (202.5, 247.5, 'rearleft', 'purple'),
        (247.5, 292.5, 'unknown (left)', 'gray'),
        (292.5, 337.5, 'frontleft', 'blue')
    ]
    
    y_max = ax.get_ylim()[1]
    
    for start, end, label, color in sectors:
        if start > end:  # Wraps around 0
            # Draw two rectangles
            ax.add_patch(Rectangle((start, 0), 360 - start, y_max, alpha=0.2, color=color))
            ax.add_patch(Rectangle((0, 0), end, y_max, alpha=0.2, color=color))
            ax.text((start + 360) / 2, y_max * 0.95, label, ha='center', va='top', fontsize=9, fontweight='bold')
        else:
            ax.add_patch(Rectangle((start, 0), end - start, y_max, alpha=0.2, color=color))
            ax.text((start + end) / 2, y_max * 0.95, label, ha='center', va='top', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('CFV Dataset Angle Distribution with Viewpoint Sector Mapping', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_html_report(csv_path: str, images_dir: str, output_path: str):
    """Generate comprehensive HTML visualization report."""
    print("Loading sample images...")
    samples = load_sample_images_per_viewpoint(csv_path, images_dir, samples_per_class=16)
    
    print("Creating viewpoint grid...")
    grid_fig = create_viewpoint_grid(samples)
    grid_base64 = fig_to_base64(grid_fig)
    
    print("Creating angle distribution plot...")
    dist_fig = create_angle_distribution_plot(csv_path)
    dist_base64 = fig_to_base64(dist_fig)
    
    # Load statistics
    from cfv_to_viewpoint import analyze_cfv_dataset
    stats = analyze_cfv_dataset(csv_path)
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CFV Dataset Viewpoint Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .viewpoint-dist {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .image-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .alert {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>🚗 CFV Dataset - 7-Class Viewpoint Analysis Report</h1>
    
    <div class="stats">
        <h2>📊 Dataset Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{stats['total_samples']:,}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['valid_samples']:,}</div>
                <div class="stat-label">Valid Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['unique_identities']}</div>
                <div class="stat-label">Unique Cars</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats['angle_range'][0]:.0f}°-{stats['angle_range'][1]:.0f}°</div>
                <div class="stat-label">Angle Range</div>
            </div>
        </div>
    </div>
    
    <div class="viewpoint-dist">
        <h2>🎯 Viewpoint Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Viewpoint</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Angle Range</th>
                </tr>
            </thead>
            <tbody>
"""
    
    viewpoint_ranges = {
        'front': '337.5° - 22.5° (0° ± 22.5°)',
        'frontright': '22.5° - 67.5° (45° ± 22.5°)',
        'rearright': '112.5° - 157.5° (135° ± 22.5°)',
        'rear': '157.5° - 202.5° (180° ± 22.5°)',
        'rearleft': '202.5° - 247.5° (225° ± 22.5°)',
        'frontleft': '292.5° - 337.5° (315° ± 22.5°)',
        'unknown': '67.5° - 112.5° & 247.5° - 292.5°'
    }
    
    total_valid = sum(stats['viewpoint_distribution'].values())
    for vp in ['front', 'frontleft', 'frontright', 'rear', 'rearleft', 'rearright', 'unknown']:
        count = stats['viewpoint_distribution'].get(vp, 0)
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        html_content += f"""
                <tr>
                    <td><strong>{vp}</strong></td>
                    <td>{count:,}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{viewpoint_ranges[vp]}</td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
    </div>
    
    <div class="alert">
        <strong>ℹ️ Note:</strong> The "unknown" class (26.5%) represents pure side views (left at 270° and right at 90°), 
        which don't align with the 6 main viewpoint classes optimized for car inspection and analysis.
    </div>
    
    <div class="image-container">
        <h2>📈 Angle Distribution with Sector Mapping</h2>
        <img src="data:image/png;base64,{dist_base64}" alt="Angle Distribution">
    </div>
    
    <div class="image-container">
        <h2>🖼️ Sample Images Per Viewpoint</h2>
        <img src="data:image/png;base64,{grid_base64}" alt="Viewpoint Grid">
    </div>
    
    <footer style="margin-top: 40px; padding: 20px; text-align: center; color: #7f8c8d; border-top: 1px solid #ddd;">
        <p>Generated by CFV Viewpoint Visualizer | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CFV viewpoint visualization report')
    parser.add_argument('--csv', type=str, default='CFV-Dataset/train.csv',
                        help='Path to CSV file')
    parser.add_argument('--images', type=str, default='CFV-Dataset/images',
                        help='Path to images directory')
    parser.add_argument('--output', type=str, default='cfv_visualization_report.html',
                        help='Output HTML file path')
    
    args = parser.parse_args()
    
    generate_html_report(args.csv, args.images, args.output)
