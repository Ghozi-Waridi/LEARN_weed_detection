import numpy as np
class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0, keepdims=True)

    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def backward_relu(grad_output, relu_input):
        return grad_output * (relu_input > 0).astype(float)
    
    @staticmethod
    def backward_sigmoid(grad_output, sig_output):
        return grad_output * sig_output * (1 - sig_output)
    
    @staticmethod
    def backward_tanh(x):
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def backward_softmax(x):
        sof = ActivationFunctions.softmax(x).reshape(-1, 1)
        return np.diagflat(sof) - np.dot(sof, sof.T)
