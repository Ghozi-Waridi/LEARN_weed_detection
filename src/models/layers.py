import numpy as np
from src.models.forward import Forward
from src.models.backward import Backward
from src.models.activations import ActivationFunctions

class Layers:
    def forward_layer(x, kernel, stride=1, pool_size=(2,2), pool_stride=2):
        """
        Forward pass through the convolutional layer, max pooling layer, flattening, and fully connected layer.
        
        Parameters:
        - x: Input image.
        - y: Weights for the fully connected layer.
        - kernel: Kernel for the convolutional layer.
        - stride: Stride for the convolutional layer.
        - pool_size: Size of the pooling window.
        - pool_stride: Stride for the pooling operation.
        
        Returns:
        - Output after passing through all layers.
        """
        
        # Convolutional Layer
        conv_output = Forward.conv_layer(x, kernel, stride)
        
        # Relu Activation
        relu_output = ActivationFunctions.relu(conv_output)
        
        # Mac_pooling Layer
        pool_output = Forward.maxPooling(relu_output, pool_size=pool_size, stride=pool_stride)
        
        # Relu Activation
        relu_output = ActivationFunctions.relu(pool_output)
        
        # Flattening
        flat_output = Forward.flatten(relu_output)
        
        # Relu Activation
        relu_output = ActivationFunctions.relu(flat_output)
        
        # # Fully Connected Layer
        # fc_output = Forward.fully_connected(relu_output, weights=weight, bias=bias)
        
        # # Sigmoid Activation
        # sigmoid_output = ActivationFunctions.sigmoid(fc_output)
        
        return relu_output