import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from pnebulae_torch.normalize import TypicalImageNorm


class NebulaeDataset(Dataset):
    """
    Clase para cargar un conjunto de datos de nebulosas.
    
    Parámetros:
        image_path (str): Ruta a las imágenes.
        mask_path (str): Ruta a las máscaras.
        dataframe (DataFrame): DataFrame con los nombres de las imágenes y máscaras.
        rsize (tuple, opcional): Tamaño para usar en la transformación de redimensionamiento predeterminada.
        transform (callable, opcional): Transformaciones opcionales para aplicar.
    
    Atributos:
        image_path (str): Ruta a las imágenes.
        mask_path (str): Ruta a las máscaras.
        data_dict (dict): Diccionario con los nombres de las imágenes y máscaras.
        img_files (list): Rutas a las imágenes.
        mask_files (list): Rutas a las máscaras.
        names (list): Nombres de las imágenes y máscaras.
        rsize (tuple): Tamaño al que redimensionar las imágenes.
        transform (torchvision.transforms): Transformaciones a aplicar a las imágenes y máscaras.
    """
    
    def __init__(self, image_path, mask_path, dataframe, rsize = None, transform = None):
        super().__init__()
        
        self.image_path = image_path  # Ruta a las imágenes
        self.mask_path = mask_path  # Ruta a las máscaras
        
        # Cargar los nombres de las imágenes y máscaras desde el dataframe
        self.data_dict = dataframe.set_index('name').to_dict(orient='index')

        # Filtrar las rutas de archivo según los nombres en el dataframe
        self.img_files = [os.path.join(self.image_path, files['h']) for files in self.data_dict.values()]
        self.mask_files = [os.path.join(self.mask_path, files['mask']) for files in self.data_dict.values()]
        self.names = list(self.data_dict.keys())  # Nombres de las imágenes y máscaras
                
        self.rsize = rsize  # Size to use in default Resize transform
        self.transform = transform

    # Returns both the image and the mask
    def __getitem__(self, index):
        """
        Obtiene la imagen y la máscara en el índice especificado.

        Args:
            index (int): Índice del elemento a obtener.

        Returns:
            tuple: Una tupla que contiene la imagen y la máscara.
        """
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        
        image = np.flip(fits.getdata(img_path, memmap=False).astype(np.float32), axis=0)
        mask = plt.imread(mask_path)
        
        # Take only the first channel. CHANGE THIS IF WE ARE GOING TO WORK WITH NUMEROUS CHANNELS
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        if len(image.shape) > 2:
            image = image[:,:,0]
        
        # Apply the defined transformations to both image and mask
        if self.transform is not None:
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            rd.seed(seed) # apply this seed to image transforms
            torch.manual_seed(seed)
            if type(self.transform) == tuple:
                image = self.transform[0](image)
            else:
                image = self.transform(image)
            rd.seed(seed) # apply the same seed to mask transforms
            torch.manual_seed(seed) 
            if type(self.transform) == tuple:
                mask = self.transform[1](mask)
            else:
                mask = self.transform(mask)
        else:
            if self.rsize is not None:
                t = transforms.Compose([
                    TypicalImageNorm(),
                    transforms.ToTensor(),
                    transforms.Resize(self.rsize, interpolation= InterpolationMode.NEAREST)
                    ])
            else:
                t = transforms.Compose([
                    TypicalImageNorm,
                    transforms.ToTensor()
                    ])

            image = t(image)
            mask = t(mask)
        
        return image, torch.round(mask)

    def __len__(self):
        """
        Obtiene la longitud del conjunto de datos.

        Returns:
            int: La longitud del conjunto de datos
        """
        return len(self.img_files)
    
    def plot(self, index, plot_image = True, plot_mask = False):
        """
        Muestra una imagen y/o máscara aleatoria del lote.
        
        Args:
            index (int): Índice del lote.
            plot_image (bool, opcional): Si es True, muestra la imagen. Por defecto es True.
            plot_mask (bool, opcional): Si es True, muestra la máscara. Por defecto es False.
        """
        image, mask = self[index]
        image = image.permute(1,2,0)
        mask = mask.permute(1,2,0)
        name = self.names[index]
        
        if plot_image:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            fig.suptitle(f"Canales de la nebulosa {name}", fontweight = 'bold', fontsize = 14)
            ax.imshow(image, cmap = "gray")
            ax.set_title(f"Canal H")
            fig.show()
        if plot_mask:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            fig.suptitle(f"Máscara de la nebulosa {name}", fontweight = 'bold', fontsize = 14)
            ax.imshow(mask, cmap = "gray")
            fig.show()
            
    def different_shapes(self):
        """
        Obtiene una lista de las diferentes formas de las imágenes en el conjunto de datos.

        Returns:
            list: Una lista de las diferentes formas de las imágenes.
        """
        shapes = set([tuple(self[i][0].permute(2,1,0).shape) for i in range(len(self))])
        return list(shapes)
    
    def bg_obj_proportions(self):
        """
        Calcula las proporciones de fondo y objeto en las máscaras.
        
        Returns:
            list: Una lista de las proporciones de fondo y objeto.
        """
        proportions = []
        for i in range(len(self)):
            mask = self[i][1].numpy()
            bg = np.sum(mask == 0)
            obj = np.sum(mask == 1)
            proportions.append(obj/(bg+obj))
        return proportions
    
    def contrast_differences(self, radius = None):
        """
        Calcula las diferencias de contraste entre el fondo y el objeto.

        Args:
            radius (int, opcional): Radio para calcular el contraste. Si no se proporciona, se calcula el contraste para toda la imagen.

        Returns:
            list: Una lista de las diferencias de contraste.
        """
        contrasts = []
        if radius is not None:
            for i in range(len(self)):
                image = self[i][0].numpy()
                mask = self[i][1].numpy()
                bg = image[mask == 0]
                obj = image[mask == 1]
                contrasts.append(np.mean(obj) - np.mean(bg))
        else:
            for i in range(len(self)):
                image = self[i][0].numpy()
                mask = self[i][1].numpy()
                
                bg = image[mask == 0]
                obj = image[mask == 1]
                contrasts.append(np.mean(obj) - np.mean(bg))
        return contrasts