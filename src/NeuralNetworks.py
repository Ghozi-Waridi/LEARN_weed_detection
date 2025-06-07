from _typeshed import SliceableBuffer
import numpy as np
import os 
import sys
import tqdm
import cv2
import matplotlib.pyplot as plt
from src.models.layers import Layers
class NeuralNetwork:
    def __init__(self, input_shape, feature_maps, num_classes, kernel, kernel_size=(3,3), pool_size=(2,2), pool_stride=2, stride=1):

        self.input_size = input_shape
        self.hidden_size = feature_maps 
        self.output_size = num_classes
        self.kernel_size = kernel_size

        '''
        Convulutional Layer Output
        '''
        self.conv_output_size = (
            (self.input_size[0] - self.kernel_size[0]) // stride + 1,
            (self.input_size[1] - self.kernel_size[1]) // stride + 1
        )

        '''
        Pooling Layer Output
        ''' 
        self.pool_output_size = ( 
            (self.conv_output_size[0] - pool_size[0]) // pool_stride + 1,
            (self.conv_output_size[1] - pool_size[1]) // pool_stride + 1
        )
        '''
        Flatter Layer Output
        '''
        self.flatten_size = self.pool_output_size[0] * self.pool_output_size[1] * self.hidden_size

        '''
        Weight anda Bias Initialization
        '''
        self.W1 = np.random.randn(self.flatten_size, self.hidden_size) * 0.01
        self.B1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.B2 = np.zeros((1, self.output_size))
        
        

    def forward(self, x):
         

        

         

        
        
        
