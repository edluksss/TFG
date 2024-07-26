import numpy as np
from skimage import morphology, exposure
from scipy import ndimage
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class ApplyMorphology:
    """
    Una clase para aplicar operaciones de morfología a imágenes.
    """
    def __init__(self, operation = morphology.opening, concat = False, **kwargs):
        """
        Construye el objeto ApplyMorphology con la operación, la concatenación y los argumentos dados.

        Parámetros
        ----------
            operation (function) : La operación de morfología a aplicar (por defecto es morphology.opening).
            concat (bool) : Si es True, concatena la imagen original y la imagen filtrada (por defecto es False).
            kwargs (dict) : Argumentos adicionales para la operación de morfología.
        """
        self.concat = concat
        self.operation = operation
        self.kwargs = kwargs
        if operation == morphology.binary_opening or operation == morphology.binary_closing:
            self.mode = "star_background"
        else:
            self.mode = "nebulae"
    
    def __call__(self, im):
        """
        Aplica la operación de morfología a la imagen dada.

        Parámetros
        ----------
            im : np.array
                La imagen a la que se aplicará la operación de morfología.

        Devoluciones
        -------
            np.array
                La imagen después de aplicar la operación de morfología.
        """
        im_orig = im.copy()
        if len(im.shape) == 3 and im.shape[2] > 1:
            im = im[:,:,-1]
        
        if self.mode == "nebulae":
            im_filt = self.operation(im, **self.kwargs)
        else:
            im_preproc = np.copy(im)
            im_filt = ndimage.gaussian_filter(im, sigma=3)
            im_filt[im == 0] = 0

            im_zonas_claras_peq = im > (im_filt + np.std(im))

            im_zonas_claras_peq = self.operation(im_zonas_claras_peq, **self.kwargs)
                
            im_preproc = (im_preproc - np.min(im_preproc))
            im_preproc[im_zonas_claras_peq] = 0
            
            im_filt = im_preproc
        if self.concat:
            if len(im_orig.shape) < 3:
                im_orig = np.expand_dims(im_orig, axis=2)
                
            im_filt = np.expand_dims(im_filt, axis=2)
            return np.concatenate((im_orig, im_filt), axis=2)
        else:
            return self.operation(im, **self.kwargs)

class ApplyIntensityTransformation:
    """
    Una clase para aplicar transformaciones de intensidad a imágenes.
    """
    def __init__(self, transformation = exposure.rescale_intensity, concat = False, **kwargs):
        """
        Construye el objeto ApplyIntensityTransformation con la transformación, la concatenación y los argumentos dados.

        Args:
            transformation (function, optional): La transformación de intensidad a aplicar. Defaults to exposure.rescale_intensity.
            concat (bool, optional):  Si es True, concatena la imagen original y la imagen transformada. Defaults to False.
            **kwargs: Argumentos adicionales para la transformación de intensidad.
            
        """
        self.transformation = transformation
        self.kwargs = kwargs
        self.concat = concat
        self.in_range = None
        self.kernel_size = None
        
        if "in_range" in self.kwargs:
            self.in_range = self.kwargs["in_range"]
        
        if "kernel_size" in self.kwargs:
            self.kernel_size = self.kwargs["kernel_size"]
    
    def __call__(self, im):
        """
        Aplica la transformación de intensidad a la imagen dada.

        Args:
            im (np.array): La imagen a la que se aplicará la transformación de intensidad.

        Returns:
            np.array: La imagen después de aplicar la transformación de intensidad.
        """
        im_orig = im.copy()
        if len(im.shape) == 3 and im.shape[2] > 1:
            im = im[:,:,-1]
        
        if self.in_range is not None:
            self.kwargs["in_range"] = (im.max() * self.in_range[0], im.max() * self.in_range[1])
        
        # self.kwargs["in_range"] = (im.min(), im.max()) # Linea para realizar un reescalado de la intensidad de la imagen lineal
        
        if self.kernel_size is not None:
            self.kwargs["kernel_size"] = im.shape[0] // self.kernel_size
            
        im_trans = self.transformation(im, **self.kwargs)
        if self.concat:
            if len(im_orig.shape) < 3:
                im_orig = np.expand_dims(im_orig, axis=2)
                
            im_trans = np.expand_dims(im_trans, axis=2)
            return np.concatenate((im_orig, im_trans), axis=2)
        else:
            return self.transformation(im, **self.kwargs)
        
class ApplyFilter:
    """
    Una clase para aplicar filtros a imágenes.
    """
    def __init__(self, filter = ndimage.gaussian_filter, concat = False, **kwargs):
        """
        Construye el objeto ApplyFilter con el filtro, la concatenación y los argumentos dados.

        Args:
            filter (function, optional): El filtro a aplicar. Defaults to ndimage.gaussian_filter.
            concat (bool, optional):  Si es True, concatena la imagen original y la imagen filtrada. Defaults to False.
            **kwargs: Argumentos adicionales para el filtro.
        """
        self.filter = filter
        self.kwargs = kwargs
        self.concat = concat
    
    def __call__(self, im):
        """
        Aplica el filtro a la imagen dada.

        Args:
            im (np.array): La imagen a la que se aplicará el filtro.

        Returns:
            np.array: La imagen después de aplicar el filtro.
        """
        im_orig = im.copy()
        if len(im.shape) == 3 and im.shape[2] > 1:
            im = im[:,:,-1]
        
        im_filt = self.filter(im, **self.kwargs)
        if self.concat:
            if len(im_orig.shape) < 3:
                im_orig = np.expand_dims(im_orig, axis=2)
                
            im_filt = np.expand_dims(im_filt, axis=2)
            return np.concatenate((im_orig, im_filt), axis=2)
        else:
            return self.filter(im, **self.kwargs)
        
class CustomPad():
    """
    Una clase para aplicar un relleno personalizado a las imágenes.
    """
    def __init__(self, target_size = (980, 980), fill = 0, fill_min = False, tensor_type = None):
        """
        Construye el objeto CustomPad con el tamaño objetivo, el valor de relleno, la opción de relleno mínimo y el tipo de tensor dados.

        Args:
            target_size (tuple, optional): El tamaño objetivo para el relleno. Defaults to (980, 980).
            fill (int, optional): El valor de relleno. Defaults to 0.
            fill_min (bool, optional): Si es True, el valor de relleno será el mínimo de la imagen. Defaults to False.
            tensor_type (type, optional): El tipo de tensor para la imagen de salida. Defaults to None.
        """
        self.target_size = target_size
        self.fill = fill
        self.fill_min = fill_min
        self.tensor_type = tensor_type
        
    def __call__(self, image):
        """
        Aplica el relleno a la imagen dada.

        Args:
            image (np.array): La imagen a la que se aplicará el relleno.

        Returns:
            np.array: La imagen después de aplicar el relleno.
        """
        # Get the size of the input image
        width, height = image.shape[2], image.shape[1]

        if width == self.target_size[1] and height == self.target_size[0]:
            if self.tensor_type is not None:
                return self.tensor_type(image)
            else:
                return image
        
        elif width > self.target_size[1] or height > self.target_size[0]:
            image = transforms.functional.resize(image, self.target_size, interpolation=InterpolationMode.NEAREST)
            if self.tensor_type is not None:
                return self.tensor_type(image)
            else:
                return image
        
        else:
            # Compute the size of the padding
            pad_width = self.target_size[1] - width
            pad_height = self.target_size[0] - height

            # Compute the padding
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top

            if self.fill_min:
                self.fill = image.min().item()
                
            # Apply the padding
            if self.tensor_type is not None:
                return self.tensor_type(transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill = self.fill))
            else:
                return transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill = self.fill)
            
class CutValues:
    """
    Una clase para cortar los valores que se consideren atípicos de una imagen.
    """
    def __init__(self, factor = 10):
        """ Construye el objeto CutValues con el factor multiplicativo de la desviación estandar, que va a definir el umbral para considerar valores atípicos.

        Args:
            factor (int, optional): factor multiplicativo de la desviación estandar. Defaults to 10.
        """
        
        self.factor = factor
        
    def __call__(self, x):
        """Corta los valores que se consideren atípicos de la imagen dada.

        Args:
            x (np.array): Imagen de entrada

        Returns:
            np.array: Imagen después de cortar los valores atípicos.
        """
        x_mean = x.mean()
        x_std = x.std()

        x[x > x_mean + self.factor * x_std] = x_mean + self.factor * x_std
        x[x < x_mean - self.factor * x_std] = x_mean - self.factor * x_std
        
        return x
    def __repr__(self):
        
        return self.__class__.__name__ + '()'