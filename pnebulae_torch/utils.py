import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
    
class DivideWindowsSubset(Dataset):
    def __init__(self, subset, window_shape = 128, fill = 0, fill_min = False):
        self.subset = subset
        
        self.x_windows, self.y_windows = [], []
        pad_width, pad_height = 0, 0
        for x, y in subset:
            
            if fill_min:
                fill = x.min().item()
                
            if x.shape[1] % window_shape != 0:
                pad_height = window_shape - (x.shape[1] % window_shape)

            if x.shape[2] % window_shape != 0:
                pad_width = window_shape - (x.shape[2] % window_shape)
                
            # Compute the padding
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            

            x_pad = transforms.functional.pad(x, (pad_left, pad_top, pad_right, pad_bottom), fill = fill)
            y_pad = transforms.functional.pad(y, (pad_left, pad_top, pad_right, pad_bottom), fill = 0)
            # y_pad = transforms.functional.pad(y, (pad_left, pad_top, pad_right, pad_bottom), fill = -1)

            x_unfold = x_pad.unfold(1, window_shape, window_shape).unfold(2, window_shape, window_shape).reshape(-1, x_pad.shape[0], window_shape, window_shape)
            y_unfold = y_pad.unfold(1, window_shape, window_shape).unfold(2, window_shape, window_shape).reshape(-1, y_pad.shape[0], window_shape, window_shape)
            
            self.x_windows.append(x_unfold)
            self.y_windows.append(y_unfold)
        
        self.x_windows = torch.cat(self.x_windows, dim = 0)
        self.y_windows = torch.cat(self.y_windows, dim = 0)
                
    def __getitem__(self, index):
        x = self.x_windows[index]
        y = self.y_windows[index]
        return x, y

    def __len__(self):
        return len(self.x_windows)