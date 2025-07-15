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
from model.swin_transformer_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnetLP(nn.Module):
    def __init__(self, config, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnetLP, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.swin_unet = SwinTransformerSys(img_size=config['--img-size'],# default val: 224
                                patch_size=config['--swin-patch-size'],#default
                                in_chans=config['--model-swin-in-chans'], #.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config['--model-swin-embed-dim'],
                                depths=config['--model-swin-depths'],
                                num_heads=config['--model-swin-num-heads'],
                                window_size=config['--model-swin-window-size'],
                                mlp_ratio=config['--model-swin-mlp-ratio'],
                                qkv_bias=config['--model-swin-qkv-bias'],
                                qk_scale=config['--model-swin-qk-scale'],
                                drop_rate=config['--model-drop-rate'],
                                drop_path_rate=config['--model-drop-path-rate'],
                                ape=config['--model-swin-ape'],
                                patch_norm=config['--model-swin-patch-norm'],
                                use_checkpoint=config['--train-use-checkpoint'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config['model_pretrain_ckpt'] #MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
"""
if __name__=='__main()__':
    swintest=SwinUnetLP()
    swintest.load_from()
"""
