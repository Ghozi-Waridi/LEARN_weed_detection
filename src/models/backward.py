import os 
import numpy as np
class Backward:

    @staticmethod
    def grad_fully_connected(grad_output, input_data, weights):
        """
        Mnehitung Gradient dari Fully Connected Layer
        """
        grad_input = np.dot(grad_output, weights.T)
        grad_weights = np.dot(input_data.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        return grad_input, grad_weights, grad_bias

    @staticmethod
    def grad_flatten(grad_output, input_shape):
        """
        Menghitung Gradient dari Flatten Layer
        """
        return grad_output.reshape(input_shape)

    @staticmethod
    def grad_maxPooling(grad_output, image, pool_size=(2,2)):
        """
        Menghitung Gradient dari MaxPooling Layer
        """
        num_maps, out_w, out_h = grad_output.shape
        _, x, y = image.shape
        grad_input = np.zeros_like(image)

        for k in range(num_maps):
            for i in range(out_w):
                for j in range(out_h):
                    point_w = i * pool_size[0]
                    point_h = j * pool_size[1]
                    patch = image[k, point_w:point_w + pool_size[0], point_h:point_h + pool_size[1]]
                    max_index = np.argmax(patch)
                    max_row, max_col = np.unravel_index(max_index, pool_size)
                    grad_input[k, point_w + max_row, point_h + max_col] = grad_output[k, i, j]

        return grad_input

    @staticmethod
    def grad_conv_layer(grad_output, image, kernels, stride=1):
        """
        Menghitung Grdient dari Convolutional Layer
        """
        image= np.array(image)
        x, y = image.shape
        num_kernels, x_kernel, y_kernel = kernels.shape
        out_w, out_h = grad_output.shape[1], grad_output.shape[2]

        grad_input = np.zeros_like(image)
        grad_kernels = np.zeros_like(kernels)
        for k in range(num_kernels):
            for i in range(out_w):
                for j in range(out_h):
                    point_h = i * stride
                    point_w = j * stride
                    patch = image[point_h:point_h + x_kernel, point_w:point_w + y_kernel]
                    grad_kernels[k, :, :] += grad_output[k, i, j] * patch
                    grad_input[point_w:point_w + x_kernel, point_h:point_h + y_kernel] = np.sum(np.dot( grad_output[k,i,j], kernels[k, :, :]))
        return grad_input, grad_kernels

