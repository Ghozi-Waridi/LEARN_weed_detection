import numpy as np


class Forward:        
    @staticmethod
    def conv_layer( image, kernel, stride=1):
        image  = np.array(image)
        x, y = image.shape
            
        kernel = np.array(kernel)
        x_kernel, y_kernel = kernel.shape
            
        output = np.zeros(((x - x_kernel) // stride + 1, (y - y_kernel) // stride + 1))
    ## Membuat sebuah ukuran output dengan menyesuaikan dengan ukuran kernel dan berapa langkah yang akan di ambil
        out_h = output.shape[1] # kita mengambil tinggi dari output untuk melakukan perulangan dan supaya sesuai dengna ukuran hasil dari output nantinnya
        out_w = output.shape[0]
            
        for i in range(out_w):
            for j  in range(out_h):
                point_h = i * stride
                point_w = j * stride
                patch = image[point_h:point_h + kernel.shape[0], point_w:point_w + kernel.shape[1]] ## Gambar yang di ambil sesuai dengan ukuran kernel unutk melakuakn operasi convulation
                output[i, j] += np.sum(np.dot(patch, kernel)) 
                    
        return output

    @staticmethod
    def maxPooling( image, pool_size=(2, 2), stride=2):
        image = np.array(image)
        x, y = image.shape
            
        output = np.zeros(((x - pool_size[0]) // stride + 1, (y - pool_size[1]) // stride + 1))
        out_h = output.shape[0]
        out_w = output.shape[1]
            
        for i in range(out_w):
            for j in range(out_h):
                point_w = i * stride
                point_h = j * stride
                patch = image[point_w:point_w + pool_size[0], point_h:point_h + pool_size[1]]
                output[i, j] = np.max(patch)
                    
        return output
    
    @staticmethod
    def flatten( image):
        return image.flatten()
     
    @staticmethod
    def fully_connected( image, weights, bias):
        image = np.array(image)
        result = np.dot(image, weights) + bias
        return result
               
                
                
        