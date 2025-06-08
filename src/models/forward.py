import numpy as np


class Forward:        
    @staticmethod
    def conv_layer( image, kernels, stride=1):
        image  = np.array(image)
        x, y = image.shape
            
        kernel = np.array(kernels)
        nums_kernel, x_kernel, y_kernel = kernel.shape
            
        output = np.zeros(((x - x_kernel) // stride + 1, (y - y_kernel) // stride + 1))
    ## Membuat sebuah ukuran output dengan menyesuaikan dengan ukuran kernel dan berapa langkah yang akan di ambil
        out_h = output.shape[1] # kita mengambil tinggi dari output untuk melakukan perulangan dan supaya sesuai dengna ukuran hasil dari output nantinnya
        out_w = output.shape[0]
        
        feature_maps = np.zeros((nums_kernel, out_w, out_h))
        for k in range(nums_kernel):
            kernel = kernels[k,:,:]
            for i in range(out_w):
                for j  in range(out_h):
                    point_h = i * stride
                    point_w = j * stride
                    patch = image[point_h:point_h + x_kernel, point_w:point_w + y_kernel] ## Gambar yang di ambil sesuai dengan ukuran kernel unutk melakuakn operasi convulation
                    feature_maps[k,i,j] = np.sum(np.dot(patch, kernel)) 

                    
        return feature_maps

    @staticmethod
    def maxPooling( image, pool_size=(2, 2), stride=2):
        image = np.array(image)
              
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        num_maps, x, y = image.shape

        output = np.zeros(((x - pool_size[0]) // stride + 1, (y - pool_size[1]) // stride + 1))
        out_h = output.shape[0]
        out_w = output.shape[1]
        
        pooled_maps = np.zeros((num_maps, out_w, out_h))
        for k in range(num_maps):
            for i in range(out_w):
                for j in range(out_h):
                    point_w = i * stride
                    point_h = j * stride
                    patch = image[k, point_w:point_w + pool_size[0], point_h:point_h + pool_size[1]]
                    pooled_maps[k,i,j] = np.max(patch)
                    
        return pooled_maps
    
    @staticmethod
    def flatten( image):
         return image.reshape(1, -1)    

    @staticmethod
    def fully_connected( image, weights, bias):
        image = np.array(image)
        result = np.dot(image, weights) + bias
        return result
               
                
                
        
