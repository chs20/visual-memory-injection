import os
import glob
import pandas as pd
import numpy as np
import json
from utils.config import CONFIG



def load_google_landmarks_dataset(n_data=None):
    """Load landmarks dataset with metadata from CSV and subsample for reproducibility.
    
    Args:
        n_data: Number of images to subsample (if None, return all)
        
    Returns:
        List of dicts with keys: image_path, image_id, landmark_name, city
    """
    random_seed = 0
    csv_path = "/mnt/cschlarmann37/data/google-landmark/index/index_image_to_name_city_v1.csv"
    base_dir = "/mnt/cschlarmann37/data/google-landmark/index"
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} landmark entries from CSV")
    
    # Build dataset from CSV entries
    dataset_images = []
    for _, row in df.iterrows():
        image_id = row['id']
        landmark_name = row['landmark_name']
        city = row['city']
        
        # Construct image path based on directory structure
        image_path = os.path.join(base_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.jpg")
        
        # Check if image file actually exists
        if os.path.exists(image_path):
            dataset_images.append({
                'image_path': image_path,
                'image_id': image_id,
                'landmark_name': landmark_name,
                'city': city
            })
        else:
            print(f"Warning: Image file not found for ID {image_id}, skipping")
    
    print(f"Found {len(dataset_images)} images with valid files")
    
    # Subsample with fixed random seed for reproducibility
    if n_data is not None and n_data < len(dataset_images):
        generator = np.random.default_rng(random_seed)
        indices = generator.choice(len(dataset_images), size=n_data, replace=False)
        dataset_images = [dataset_images[i] for i in indices]
        print(f"Subsampled to {len(dataset_images)} images using random seed {random_seed}")
    
    return dataset_images

def load_handselected_landmarks_dataset():
    """Load our hand-selected landmarks dataset from clean_images directory.
    
    Args:
        data_path: Path to the dataset json file
        
    Returns:
        List of dicts with keys: image_path, image_id, landmark_name, city
    """
    data_path = CONFIG.handselected_landmarks_data_path
    base_dir = os.path.dirname(data_path)
    images_dir = os.path.join(base_dir, "images")
    dataset_images = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    for image in data['images']:
        image_path = os.path.join(images_dir, image['file_name'])
        dataset_images.append({
            'image_path': image_path,
            'landmark_name': image['landmark_name'],
            'city': image['city'],
            'license': image['license']
        })
    return dataset_images

def load_coco_dataset(dataset_name):
    """Load COCO dataset from clean_images directory.
    
    Args:
        dataset_name: Name of COCO dataset (e.g., 'coco20', 'coco100', 'coco500')
        
    Returns:
        List of dicts with keys: image_path, image_id
    """
    base_dir = f"./clean_images/{dataset_name}"
    
    if not os.path.exists(base_dir):
        raise ValueError(f"COCO dataset directory not found: {base_dir}")
    
    # Find all images in the directory
    image_pattern = os.path.join(base_dir, "*.png")
    all_images = glob.glob(image_pattern)
    print(f"Found {len(all_images)} images in {dataset_name}")
    
    # Build dataset from found images
    dataset_images = []
    for image_path in all_images:
        # Extract image ID from filename
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        dataset_images.append({
            'image_path': image_path,
            'image_id': image_id,
            'landmark_name': None,  # No landmark data for COCO
            'city': None
        })

    # sort by image_id
    dataset_images.sort(key=lambda x: x['image_id'])
    
    return dataset_images