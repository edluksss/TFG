import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz

class ApplyKMeans:
    def __init__(self, concat = False, **kwargs):
        self.concat = concat
        self.kwargs = kwargs

    def __call__(self, im):
        im_orig = im.copy()
        if len(im.shape) == 3 and im.shape[2] > 1:
            im = im[:,:,-1]
            
        im_array = im.reshape(-1, 1)
    
        kmeans = KMeans(**self.kwargs).fit(im_array)  # Entrenar el modelo K-Means
        
        # Obtener la imagen segmentada aplicando el algoritmo a cada píxel de la imagen
        im_segm_array = kmeans.predict(im_array)

        # Reemplazar los índices de los clústeres por los centroides de los clústeres
        im_segm_array = np.array([kmeans.cluster_centers_[i] for i in im_segm_array])

        # Cambiar las dimensiones de los datos segmentados para que se correspondan con la imagen inicial
        im_segm = im_segm_array.reshape(im.shape[0], im.shape[1], 1)
        
        if self.concat:
            if len(im_orig.shape) < 3:
                im_orig = np.expand_dims(im_orig, axis=2)
                
            return np.concatenate((im_orig, im_segm), axis=2)
        else:
            return im_segm

class ApplyFCM:
    def __init__(self, concat = False, **kwargs):
        self.concat = concat
        self.kwargs = kwargs

    def __call__(self, im):
        im_orig = im.copy()
        if len(im.shape) == 3 and im.shape[2] > 1:
            im = im[:,:,-1]
            
        im_array = im.reshape(1, -1)

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=im_array, **self.kwargs)  # Aplicar el algoritmo FCM
        
        # Asociar a cada píxel el cluster para el que tiene una mayor pertenencia
        clusters_array = np.argmax(u, axis=0)
        maximos = np.max(u, axis=0)

        # Reemplazar los índices de los clústeres por los centroides de los clústeres
        im_segm_array = np.array([cntr[i] for i in clusters_array])

        # Cambiar las dimensiones de los datos segmentados para que se correspondan con la imagen inicial
        im_segm = im_segm_array.reshape(im.shape[0], im.shape[1], 1)
        
        maximos = maximos.reshape(im.shape[0], im.shape[1], 1)
        im_segm = np.concatenate((maximos, im_segm), axis=2)
            
        if self.concat:
            if len(im_orig.shape) < 3:
                im_orig = np.expand_dims(im_orig, axis=2)
                
            return np.concatenate((im_orig, im_segm), axis=2)
        else:
            return im_segm