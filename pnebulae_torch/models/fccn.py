import torch
import torch.nn as nn
import separableconv.nn as separable_nn
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as L
from torch.optim import Adam
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from skimage import morphology
from pnebulae_torch.utils import plot_all

def conv_layer(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, **kwargs),
        nn.ReLU(inplace=False),
    )   
    
def conv_layer_transpose(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
        nn.ReLU(inplace=False),
    )   
    
def conv_layer_separable(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        separable_nn.SeparableConv2d(in_channels, out_channels, **kwargs),
        nn.ReLU(inplace=False),
    )
    
class ConvNet(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dims, output_dim, transposeConv = False, separable_conv = False, **kwargs):
        super().__init__()
        
        if separable_conv:
            conv_layer_fnc = conv_layer_separable
        else:
            conv_layer_fnc = conv_layer
            
        self.layer1_conv = conv_layer_fnc(input_dim, hidden_dims[0], **kwargs)
        
        self.hidden_layers = nn.ModuleList([conv_layer_fnc(hidden_dims[i], hidden_dims[i+1], **kwargs) for i in range(len(hidden_dims)-1)])
        
        self.hidden_layers_deconv = None
        
        if transposeConv:
            # Capas de deconvolución para aumentar la resolución de las características
            self.hidden_layers_deconv = nn.ModuleList([conv_layer_transpose(hidden_dims[i], hidden_dims[i-1], **kwargs) for i in range(len(hidden_dims)-1, 0, -1)])
        
            self.layer1_deconv = nn.ConvTranspose2d(hidden_dims[0], input_dim, **kwargs)
        
            self.output_layer = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        
        else:
            self.output_layer = nn.Conv2d(hidden_dims[-1], output_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.layer1_conv(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        if self.hidden_layers_deconv is not None:
            for layer in self.hidden_layers_deconv:
                x = layer(x)
    
            x = self.layer1_deconv(x)
        
        x = self.output_layer(x)
        
        return x
