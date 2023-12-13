# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from model.gunet import UNet 

logger = logging.getLogger(__name__)

class MetaUnet(torch.nn.Module):
    def __init__(self,pretainedfl=None):
        super(MetaUnet, self).__init__()
        if pretainedfl is None: 
          pretainedfl="/geosampling/metasegement/test12/state-1.pt"
        datamodel=torch.load(pretainedfl)
        self.optimizer_state_dict=datamodel['optimizer_state_dict']
        self.inner_lrs=datamodel['inner_lrs']
        self.meta_parameters=datamodel['meta_parameters']  
        modelconfig = {'n_channels': 3, 'n_classes': 2, 'encoders': [16, 32, 64, 128, 256, 512,1024],
                    'bilinear': True}
        self.majormodule=UNet(**modelconfig)
        self.majormodule.load_state_dict(self.meta_parameters)
        self.lp=torch.nn.Linear(2,1)
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x = self.majormodule(x)
        x=torch.moveaxis(x, 1, -1)
        x=self.lp(x)
        logits=torch.squeeze(x)
        return logits
