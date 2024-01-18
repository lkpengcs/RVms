import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from models.smp_model import Unet


class Expert(nn.Module):
    def __init__(self, args):
        super(Expert,self).__init__()
        # self.seg_net = smp.Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', in_channels=5, classes=1, activation='sigmoid')
        self.seg_net = Unet(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', in_channels=5, classes=1, activation='sigmoid')
    
    def forward(self, x):
        return self.seg_net(x)