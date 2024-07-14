import matplotlib.pyplot as plt

def plot_all(image, mask, **kwargs):
    """
    Dibuja todos los canales de una imagen y su máscara correspondiente.

    Args:
        image (Tensor): Un tensor de PyTorch que representa la imagen. 
                    Debe tener la forma (C, H, W), donde C es el número de canales, 
                    y H y W son la altura y la anchura de la imagen, respectivamente.
        mask (Tensor): Un tensor de PyTorch que representa la máscara. 
                   Debe tener la forma (1, H, W).
        **kwargs: Argumentos adicionales para la función imshow de matplotlib.
    """

    image = image.permute(1,2,0)
    mask = mask.permute(1,2,0)
    
    n_channels = image.shape[2]
    fig, ax = plt.subplots(1, n_channels + 1, figsize=(5 * n_channels, 5))
    # fig.suptitle(f"Canales de la nebulosa y máscara", fontweight = 'bold', fontsize = 14)
    for i in range(n_channels):
        ax[i].imshow(image[:,:,i]*255, **kwargs)
        ax[i].set_title(f"Canal {i}")
        
    ax[n_channels].imshow(mask, cmap = "gray")
    ax[n_channels].set_title(f"Máscara")
    fig.show()