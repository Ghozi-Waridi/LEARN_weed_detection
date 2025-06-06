import numpy as np
import os 
import sys
import tqdm
import cv2
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_shape, num_classes, kernel_size=(3,3), pool_size=(2,2)):
        self.input_size = input_shape
        self.output_size = num_classes
        
        
        