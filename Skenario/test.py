
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt

current_path = os.getcwd()
print(current_path)
project_root_path = os.path.abspath(os.path.join(current_path, ".."))
print(project_root_path)
if project_root_path not in sys.path:
    sys.path.append(project_root_path)
from src.models.forward import Forward  


path = "CNN/Weed_detection/Dataset/ghozi.jpg"

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (64, 64))
plt.imshow(img, cmap="gray")

