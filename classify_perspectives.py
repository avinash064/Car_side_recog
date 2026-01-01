"""
Automatic Car Perspective Classification using DeepImageSearch and Clustering
Organizes car images into 6 perspective categories:
- Front, Rear, Front Left, Front Right, Rear Left, Rear Right
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

# Import DeepImageSearch for feature extraction
from DeepImageSearch import Load_Data, Search_Setup

class CarPerspectiveClassifier:
    def __init__(self, base_dir):
        """
        Initialize the classifier
        
        Args:
            base_dir: Path to car_side_recog directory
        """
        self.base_dir = Path(base_dir)
        self.traindata_dir = self.base_dir / "traindata" / "epfl-gims08" / "tripod-seq"
        self.organized_dir = self.base_dir / "organized_perspectives"
        
        # Create output directories
        self.perspective_names = [
            "front",
            "front_right",
            "front_left",
            "rear",
            "rear_right",
            "rear_left"
        ]
        
        for name in self.perspective_names:
            output_folder = self.organized_dir / name
            output_folder.mkdir(parents=True, exist_ok=True)
    
    def get_all_images(self, max_per_sequence=None):
        """
        Get all image paths from traindata
        
        Args:
            max_per_sequence: Maximum images per sequence (None for all)
            
        Returns:
            List of image paths
        """
        image_paths = []
        
        for seq_num in range(1, 21):
            seq_images = []
            for img_file in sorted(self.traindata_dir.glob(f"tripod_seq_{seq_num:02d}_*.jpg")):
                seq_images.append(img_file)
            
            if max_per_sequence:
                seq_images = seq_images[:max_per_sequence]
            
            image_paths.extend(seq_images)
        
        return image_paths
    
    def extract_features_simple(self, image_paths):
        """
        Extract simple features using ResNet50 (simpler than DeepImageSearch full pipeline)
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Feature array
        """
        print("Extracting features from images using ResNet50...")
        
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        
        # Load pretrained ResNet50
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
        model = model.to(device)
        model.eval()
        
        # Image preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        features = []
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"Processing {i + 1}/{len(image_paths)} images...")
            
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0).to(device)
                
                # Extract features
                with torch.no_grad():
                    feature = model(batch_t)
                
                features.append(feature.cpu().numpy().flatten())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add zero vector for failed images
                features.append(np.zeros(2048))
        
        return np.array(features)
    
    def cluster_images(self, features, n_clusters=6):
        """
        Cluster images based on their features
        
        Args:
            features: Feature array
            n_clusters: Number of clusters (6 for 6 perspectives)
            
        Returns:
            Cluster labels
        """
        print(f"\nClustering images into {n_clusters} perspective categories...")
        
        # Apply PCA for dimensionality reduction
        print("Applying PCA...")
        pca = PCA(n_components=min(50, features.shape[0], features.shape[1]))
        features_reduced = pca.fit_transform(features)
        
        print(f"Reduced features from {features.shape[1]} to {features_reduced.shape[1]} dimensions")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Perform K-Means clustering
        print(f"Performing K-Means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features_reduced)
        
        print(f"Clustering complete!")
        
        # Print cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} images")
        
        return labels, kmeans, pca
    
    def assign_cluster_names(self, image_paths, labels):
        """
        Assign meaningful names to clusters based on rotation patterns
        Since cars rotate around their center, we can infer perspective from sequence position
        
        Args:
            image_paths: List of image paths
            labels: Cluster labels
            
        Returns:
            Mapping of cluster_id to perspective name
        """
        # Group images by sequence and analyze cluster patterns
        sequence_clusters = {}
        
        for img_path, label in zip(image_paths, labels):
            # Extract sequence number and frame number
            filename = img_path.name
            parts = filename.replace('.jpg', '').split('_')
            seq_num = int(parts[2])
            frame_num = int(parts[3])
            
            if seq_num not in sequence_clusters:
                sequence_clusters[seq_num] = []
            
            sequence_clusters[seq_num].append((frame_num, label))
        
        # Analyze rotation patterns to determine perspective order
        # In a 360° rotation, typical order is: Front → Right → Rear → Left → Front
        # With 6 clusters, we might have: Front, Front-Right, Right/Rear-Right, Rear, Rear-Left/Left, Front-Left
        
        cluster_positions = {i: [] for i in range(6)}
        
        for seq_num, frames in sequence_clusters.items():
            frames.sort()  # Sort by frame number
            total_frames = len(frames)
            
            for frame_num, cluster_id in frames:
                # Normalize position (0.0 to 1.0 where 0.0 is start, 1.0 is end of rotation)
                position = (frame_num - 1) / max(total_frames - 1, 1)
                cluster_positions[cluster_id].append(position)
        
        # Calculate average position for each cluster
        cluster_avg_positions = {}
        for cluster_id, positions in cluster_positions.items():
            if positions:
                cluster_avg_positions[cluster_id] = np.mean(positions)
            else:
                cluster_avg_positions[cluster_id] = 0.5
        
        # Sort clusters by average position to understand rotation order
        sorted_clusters = sorted(cluster_avg_positions.items(), key=lambda x: x[1])
        
        print(f"\nCluster positions in rotation sequence:")
        for cluster_id, avg_pos in sorted_clusters:
            print(f"  Cluster {cluster_id}: average position {avg_pos:.3f} ({len(cluster_positions[cluster_id])} images)")
        
        # Map clusters to perspectives based on typical rotation
        # Assumption: Full 360° rotation goes Front → Front-Right → Rear-Right → Rear → Rear-Left → Front-Left
        perspective_order = ["front", "front_right", "rear_right", "rear", "rear_left", "front_left"]
        
        cluster_to_perspective = {}
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            cluster_to_perspective[cluster_id] = perspective_order[i % 6]
        
        print(f"\nCluster to perspective mapping:")
        for cluster_id, perspective in cluster_to_perspective.items():
            print(f"  Cluster {cluster_id} -> {perspective}")
        
        return cluster_to_perspective
    
    def organize_images(self, image_paths, labels, cluster_to_perspective):
        """
        Copy images to their respective perspective folders
        
        Args:
            image_paths: List of image paths
            labels: Cluster labels
            cluster_to_perspective: Mapping of cluster_id to perspective name
        """
        print(f"\nOrganizing images into perspective folders...")
        
        perspective_counts = {name: 0 for name in self.perspective_names}
        
        for img_path, label in zip(image_paths, labels):
            perspective = cluster_to_perspective[label]
            output_folder = self.organized_dir / perspective
            
            # Copy image to perspective folder
            dest_path = output_folder / img_path.name
            shutil.copy2(img_path, dest_path)
            
            perspective_counts[perspective] += 1
        
        print(f"\nImages organized by perspective:")
        for perspective, count in perspective_counts.items():
            print(f"  {perspective}: {count} images")
        
        print(f"\nAll images saved to: {self.organized_dir}")
    
    def run_classification(self, max_images_per_sequence=None):
        """
        Run the complete classification pipeline
        
        Args:
            max_images_per_sequence: Maximum images per sequence (None for all)
        """
        print(f"{'='*70}")
        print("Car Perspective Automatic Classification")
        print(f"{'='*70}")
        
        # Step 1: Get all images
        print(f"\nStep 1: Collecting images...")
        image_paths = self.get_all_images(max_per_sequence=max_images_per_sequence)
        print(f"Found {len(image_paths)} images to process")
        
        # Step 2: Extract features
        print(f"\nStep 2: Extracting features...")
        features = self.extract_features_simple(image_paths)
        print(f"Extracted features shape: {features.shape}")
        
        # Step 3: Cluster images
        print(f"\nStep 3: Clustering images...")
        labels, kmeans, pca = self.cluster_images(features, n_clusters=6)
        
        # Step 4: Assign cluster names based on rotation patterns
        print(f"\nStep 4: Assigning perspective labels to clusters...")
        cluster_to_perspective = self.assign_cluster_names(image_paths, labels)
        
        # Step 5: Organize images
        print(f"\nStep 5: Organizing images...")
        self.organize_images(image_paths, labels, cluster_to_perspective)
        
        print(f"\n{'='*70}")
        print("Classification Complete!")
        print(f"{'='*70}")
        print(f"\nCheck the '{self.organized_dir}' folder for organized images")
        print(f"Each subfolder contains images from that perspective:")
        for name in self.perspective_names:
            print(f"  - {name}/")


def main():
    """Main function"""
    base_dir = r"c:\Users\xghostrider\Downloads\NEw_ProJect\car_side_recog"
    
    classifier = CarPerspectiveClassifier(base_dir)
    
    # Run classification on subset first (faster for testing)
    # Use max_images_per_sequence=20 for quick test, or None for all images
    print("Running classification on sample data (20 images per sequence)...")
    print("This will take a few minutes...")
    
    classifier.run_classification(max_images_per_sequence=20)


if __name__ == "__main__":
    main()
