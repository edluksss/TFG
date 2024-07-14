import numpy as np

class TypicalImageNorm:
    """
    Una clase para normalizar imágenes de manera típica.
    
    """
    def __init__(self, factor = 1, substract = 0):
        """
        Construye el objeto TypicalImageNorm con el factor y el valor a restar dados.

        Args:
            factor : float
                Un factor de escala para la normalización (por defecto es 1).
            substract : float
                Un valor a restar después de la normalización (por defecto es 0).
        """
        self.factor = factor
        self.substract = substract
        
    def __call__(self, x):
        """
        Aplica la normalización a la imagen dada.

        Args:
            x : np.array
                La imagen a normalizar.

        Returns:
            np.array: La imagen normalizada.
        """
        return ((x - np.min(x)) / (np.max(x) - np.min(x)) - self.substract) * self.factor

    def __repr__(self):
        """
        Devuelve una representación en cadena de la clase.

        Returns:
            str: Una representación en cadena de la clase.
        """
        return self.__class__.__name__ + '()'