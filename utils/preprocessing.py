import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(image_input):
    """
    Extracts tabular features from a skin lesion image.
    Args:
        image_input (str or numpy.ndarray): Path to the image or loaded image array.
    Returns:
        numpy.ndarray: 1D array of extracted features.
    """
    # Load image if input is a path
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image at {image_input}")
    else:
        img = image_input

    # Step 1: Resize for consistent feature extraction
    img_resized = cv2.resize(img, (128, 128))
    
    # Convert color spaces
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. Image Color Statistics (Mean and Std for B, G, R)
    b, g, r = cv2.split(img_resized)
    features.extend([np.mean(b), np.std(b), np.mean(g), np.std(g), np.mean(r), np.std(r)])
    
    # 2. HSV Color Statistics (Mean and Std for H, S, V) over the image
    h, s, v = cv2.split(hsv_img)
    features.extend([np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)])
    
    # 3. Color Histograms (Flattened, 8 bins per channel B,G,R)
    for channel in (b, g, r):
        hist = cv2.calcHist([channel], [0], None, [8], [0, 256])
        features.extend(hist.flatten())
        
    # 4. Simple Edge/Texture Feature (Canny edge density)
    edges = cv2.Canny(gray_img, 100, 200)
    edge_density = np.sum(edges > 0) / (128 * 128)
    features.append(edge_density)

    return np.array(features)

def create_tabular_dataset(metadata_csv, images_dir, output_csv):
    """
    Reads HAM10000 metadata, extracts features from images, and saves to CSV.
    """
    print("Loading metadata...")
    df = pd.read_csv(metadata_csv)
    
    # Convert one-hot encoded HAM10000 labels to binary classes
    # Malignant = High Risk, Benign = Low Risk
    malignant_classes = ['MEL', 'BCC', 'AKIEC']
    
    features_list = []
    labels_list = []
    
    print(f"Extracting features from {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image']
        
        # Determine if it is malignant
        is_malignant = False
        for c in malignant_classes:
            if c in row and row[c] == 1.0:
                is_malignant = True
                break
                
        label = 1 if is_malignant else 0
        
        # In HAM10000, images might be in nested folders, but typically they end in .jpg
        # We will assume they are all pooled in the `images` directory or we find them.
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        if os.path.exists(img_path):
            try:
                features = extract_features(img_path)
                features_list.append(features)
                labels_list.append(label)
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
        else:
            print(f"Image not found: {img_path}")

    print("Saving dataset...")
    # Create feature column names
    num_features = len(features_list[0]) if features_list else 0
    feature_cols = [f"feature_{i}" for i in range(num_features)]
    
    # Create dataframe
    dataset_df = pd.DataFrame(features_list, columns=feature_cols)
    dataset_df['label'] = labels_list
    
    dataset_df.to_csv(output_csv, index=False)
    print(f"Dataset successfully saved to {output_csv}")
    print(f"Total processed samples: {len(dataset_df)}")


if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    
    METADATA_CSV = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    OUTPUT_CSV = os.path.join(DATASET_DIR, "features.csv")
    
    # Check if dataset exists before running
    if not os.path.exists(METADATA_CSV):
        print(f"[Warning] Metadata not found at {METADATA_CSV}. Please download HAM10000 dataset first.")
    else:
        create_tabular_dataset(METADATA_CSV, IMAGES_DIR, OUTPUT_CSV)
