'''
Inference File : 
> Helps you load the model for chosen model weight path 
> It visualises both binary and scribble outputs together 
> Stores in the supplied folder name 
'''

import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import ViT

# Global settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from network import * 

settings={ "expName":'test',
    "datacode":'SD',
    "learning_rate":0.005,
    "vit_model_size":"base",
    "imgsize":256,
    "patchsize":8,
    "split_size":256,
    "vit_patch_size":8,
    "encoder_freeze":"False",
    "encoder_layers":6,
    "encoder_heads":8,
    "encoder_dims":768,
    "batch_size":8,
    "num_epochs":25,
    "train_scribble":False,
    "train_binary":False,
    "vis_results":"True",
}

# Encoder settings
encoder_layers = settings['encoder_layers']
encoder_heads = settings['encoder_heads']
encoder_dim = settings['encoder_dims']

# Encoder
v = ViT(
    image_size = settings['imgsize'],
    patch_size =  settings['patchsize'],
    num_classes = 1000,
    dim = encoder_dim,
    depth = encoder_layers,
    heads = encoder_heads,
    mlp_dim = 2048)

# Full model
network = SeamFormer(encoder = v,
    decoder_dim = encoder_dim,
    decoder_depth = encoder_layers,
    decoder_heads = encoder_heads)
network.to(device)

