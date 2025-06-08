import numpy as np
import cv2
import os
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

from src.data import dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str, split: bool = False, random_state: int = 42) -> Tuple[List[str], List[str]]:
    image_paths = []
    labels = []

    try:
        class_folders = sorted(
            [f for f in os.listdir(dataset_path)
             if os.path.isdir(os.path.join(dataset_path, f))]
        )
        logger.info(f"Classes Found; {class_folders}")

        for class_name in class_folders:
            class_path = os.path.join(dataset_path, class_name)
            logger.info(f"Loading images from class: {class_path}")

            for image_file in tqdm(os.listdir(class_path), desc=f"Loading {class_name} image"):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_path, image_file)
                    image_paths.append(img_path)
                    labels.append(class_name)

        logger.info(f"Total Images Loaded: {len(image_paths)}")
        return image_paths, labels

    except Exception as e:
        logger.error(f"Error Loading dataset: {e}")
        raise


def preprocess_image(image_path: str, image_size: Tuple[int, int] = (28,28)) -> np.ndarray:
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Image not found or could not be read: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, image_size)
        return img.astype(np.float32)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def load_and_preprocess_dataset(dataset_path: str, image_size: Tuple[int, int] = (28,28), split: bool = False, random_state: int = 42):
    try:
       
        subfolders = [f.lower() for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        has_train_test = 'train' in subfolders and 'test' in subfolders

        if has_train_test:
            train_path = os.path.join(dataset_path, 'train')
            train_image_paths, train_labels = load_dataset(train_path)
            train_images, train_valid_labels = [], []
            for img_path, label in tqdm(zip(train_image_paths, train_labels), total=len(train_image_paths), desc="Preprocessing Train Images"):
                img = preprocess_image(img_path, image_size)
                if img is not None:
                    train_images.append(img)
                    train_valid_labels.append(label)
            x_train = np.array(train_images, dtype=np.float32)
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(train_valid_labels)

            test_path = os.path.join(dataset_path, 'test')
            test_image_paths, test_labels = load_dataset(test_path)
            test_images, test_valid_labels = [], []
            for img_path, label in tqdm(zip(test_image_paths, test_labels), total=len(test_image_paths), desc="Preprocessing Test Images"):
                img = preprocess_image(img_path, image_size)
                if img is not None:
                    test_images.append(img)
                    test_valid_labels.append(label)
            x_test = np.array(test_images, dtype=np.float32)
            y_test = label_encoder.transform(test_valid_labels)  

            logger.info(f"Train images: {len(x_train)}, Test images: {len(x_test)}")
            return x_train, x_test, y_train, y_test, label_encoder

        else:

            image_paths, labels = load_dataset(dataset_path)
            images, valid_labels = [], []
            for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Preprocessing Images"):
                img = preprocess_image(img_path, image_size)
                if img is not None:
                    images.append(img)
                    valid_labels.append(label)
            if not images:
                raise ValueError("No Valid Images Found")
            x = np.array(images, dtype=np.float32)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(valid_labels)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=random_state, stratify=y
            )
            logger.info(f"Train images: {len(x_train)}, Test images: {len(x_test)}")
            return x_train, x_test, y_train, y_test, label_encoder

    except Exception as e:
        logger.error(f"Error in dataset loading and preprocessing: {str(e)}")
        raise