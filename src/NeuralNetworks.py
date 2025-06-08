

import numpy as np
import os 
import sys
import tqdm
import cv2
import matplotlib.pyplot as plt

from src.models.forward import Forward
from src.models.backward import Backward
from src.models.activations import ActivationFunctions

class NeuralNetwork:
    def __init__(self, input_shape, feature_maps, num_classes, kernel,  pool_size=(2,2), pool_stride=2, stride=1):

        self.input_size = input_shape
        self.hidden_size = feature_maps 
        self.output_size = num_classes
        self.kernel_size = kernel.shape
        self.kernel = kernel
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.stride = stride

        '''
        Convulutional Layer Output
        '''
        self.conv_output_size = (
            (self.input_size[0] - self.kernel_size[0]) // self.stride + 1,
            (self.input_size[1] - self.kernel_size[1]) // self.stride + 1
        )
        print("Conv : ", self.conv_output_size)
        '''
        Pooling Layer Output
        ''' 
        self.pool_output_size = ( 
            (self.conv_output_size[0] - self.pool_size[0]) // self.pool_stride + 1,
            (self.conv_output_size[1] - self.pool_size[1]) // self.pool_stride + 1
        )
        print("Pool : ", self.pool_output_size)
        '''
        Flatter Layer Output
        '''
        self.flatten_size = self.pool_output_size[0] * self.pool_output_size[1] * self.hidden_size 
        print("Flat : ", self.flatten_size)
        '''
        Weight anda Bias Initialization
        '''
        self.W1 = np.random.randn(self.flatten_size, self.hidden_size) * 0.01
        print(self.W1.shape)
        self.B1 = np.zeros((1, self.hidden_size))
        print(self.B1.shape)
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        print(self.W2.shape)
        self.B2 = np.zeros((1, self.output_size))
        print(self.B2.shape)
 

    def forward(self, x):
        """
        Convulutional Layer
        """
        self.conv_output = Forward.conv_layer(x, self.kernel, stride=self.stride)
        print(f"Layer Conv: {self.conv_output.shape}")
        print(f"Kernel: {self.kernel.shape}")

        """
        Relu Activation
        """
        self.relu_output = ActivationFunctions.relu(self.conv_output)

        """
        MaxPooling Layer
        """
        self.pool_output = Forward.maxPooling(self.relu_output, pool_size=self.pool_size, stride=self.pool_stride)
        print(f"Layer Pooling: {self.pool_output.shape}")
        # """
        # Relu Activation
        # """
        # relu_output2 = ActivationFunctions.relu(pool_output)

        """
        Flatten Layer
        """
        self.flat_output = Forward.flatten(self.pool_output)
        print(f"Layer Flatten: {self.flat_output.shape}")

        """ 
        Fully Connected Layer
        """
        self.fc_output = np.dot(self.flat_output, self.W1) + self.B1
        print(f"Layer Fully Connected: {self.fc_output.shape}")
        """
        Relu Activation
        """
        self.relu_output3 = ActivationFunctions.relu(self.fc_output)
        
        """
        Fully Connected Layer 2
        """
        self.fc_output2 = np.dot(self.relu_output3, self.W2) + self.B2
        
        """
        Sigmoid Activation
        """ 
        self.sig_output = ActivationFunctions.sigmoid(self.fc_output2)

        return self.sig_output

    def backward(self, grad_output, x):
        """
        Backward pass Untuk layer Update Parameter
        """
        self.grad_sig = ActivationFunctions.backward_sigmoid(grad_output, self.sig_output)
        print(f"Grad Sigmoid: {self.grad_sig.shape}")

        """
        Fully Connected Layer Gradient
        """
        self.grad_fc2, self.grad_W2, self.grad_B2 = Backward.grad_fully_connected(self.grad_sig, self.relu_output3, self.W2)
        print(f"Grad Fully Connected 2: {self.grad_fc2.shape}")
        self.grad_relu3 = ActivationFunctions.backward_relu(self.grad_fc2, self.fc_output)
        print(f"Grad Relu 3: {self.grad_relu3.shape}")
        self.grad_fc, self.grad_W1, self.grad_B1 = Backward.grad_fully_connected(self.grad_relu3, self.flat_output, self.W1)
        print(f"Grad Fully Connected 1: {self.grad_fc.shape}")

        """
        Flatter Layer Gradient
        """
        self.grad_flat = Backward.grad_flatten(self.grad_fc, self.pool_output.shape)
        print(f"Grad Flatten: {self.grad_flat.shape}")

        """
        MaxPooling Layer Gradient
        """
        self.grad_pool = Backward.grad_maxPooling(self.grad_flat, self.relu_output, pool_size=self.pool_size)
        print(f"Grad Pooling: {self.grad_pool.shape}")

        """
        Convulutional Layer Gradient
        """
        self.grad_conv, self.grad_kernel = Backward.grad_conv_layer(self.grad_pool, x, self.kernel, stride=self.stride)
        print(f"Grad Convulutional Layer: {self.grad_conv.shape}")
        print(f"Grad Kernel: {self.grad_kernel.shape}")

        self.W1 -= self.grad_W1
        self.B1 -= self.grad_B1
        self.W2 -= self.grad_W2
        self.B2 -= self.grad_B2
        self.kernel -= self.grad_kernel


        


        

         

        
        
        
