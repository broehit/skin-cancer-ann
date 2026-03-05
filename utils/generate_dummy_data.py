import os
import cv2
import numpy as np
import pandas as pd

def generate():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Generate 100 dummy images
    data = []
    labels = ['mel', 'bcc', 'akiec', 'nv', 'bkl', 'df', 'vasc']
    
    print("Generating 100 dummy images for testing...")
    for i in range(100):
        img_id = f"ISIC_{i:05d}"
        
        # Create a random noisy image
        img = np.random.randint(0, 256, (150, 200, 3), dtype=np.uint8)
        
        cv2.imwrite(os.path.join(IMAGES_DIR, f"{img_id}.jpg"), img)
        
        # Assign random label
        dx = np.random.choice(labels)
        
        data.append({
            'image_id': img_id,
            'dx': dx,
            'dx_type': 'test',
            'age': np.random.randint(20, 80),
            'sex': np.random.choice(['male', 'female']),
            'localization': 'back'
        })
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATASET_DIR, "HAM10000_metadata.csv"), index=False)
    
    print("Dummy dataset generated at dataset/")
    print("You can now test the pipeline by running:")
    print("1. python utils/preprocessing.py")
    print("2. python model/train_model.py")
    print("3. python app/app.py")

if __name__ == "__main__":
    generate()
