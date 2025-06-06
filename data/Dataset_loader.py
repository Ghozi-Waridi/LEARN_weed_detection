import os
import cv2
import numpy as  np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(dataset_path, image_size=(28, 28)):
    
    images, labels = [], []
    
    class_folder = sorted([f for f in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, f))])
    print(f"Classes Found: {class_folder}")
    for class_name in class_folder:
        class_path = os.path.join(dataset_path, class_name)
        print("Loading images from class:", class_path)
        for img_file in tqdm(os.listdir(class_path), desc=f"Loading {class_name} images"):
            img_path = os.path.join(class_path, img_file)
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = cv2.imread(img_path)
                image = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(class_name)

    x = np.array(images, dtype=np.float32)
    y = LabelEncoder().fit_transform(labels)
    print(f"Total images loaded: {len(x)}")
    print(x)
    print(f"Total labels loaded: {len(y)}")
    print(y)
    
    return x, y, LabelEncoder().fit(labels)
                    

               