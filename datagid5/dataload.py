import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler 
import os 
from PIL import Image 
import random  
import re 
import math 
from glob import glob 
from pathlib import Path 

class traincls_dataset_wmixed_gmorani_pathUp(data.Dataset):
    def __init__(self, srcpath, selistDf=None,lossweight=None,conopt=None):
        super(traincls_dataset_wmixed_gmorani_pathUp, self).__init__()
        self.list=[os.path.split(f)[-1] for f in selistDf['tfl'].values]
        self.srcpath=srcpath
        self.lossweight=selistDf[lossweight].values if lossweight is not None else None
        self.conopt=conopt

    def __getitem__(self, index):
        f_path=self.srcpath + '/' +self.list[index]
        #f_path=self.list[index]
        #print('f_path:',f_path)
        assert os.path.exists(f_path)
        try:
            adata=np.load(f_path,allow_pickle=True)
            mask=adata['orgmask']
            mask=mask[None,:,:]
            img=adata['orgimg'].astype(np.float32)/256.0
            img = np.moveaxis(img, -1, 0)
        except OSError:
            return None, None, None,None
        conopt=-1 if self.conopt is None else adata[self.conopt]
        losswei=-1 if self.lossweight is None else self.lossweight[index]
        return img, mask, conopt, losswei

    def __len__(self):
        return len(self.list)
      
      

class traincls_dataset_wmixed_gmorani_pathUp2(data.Dataset):
    def __init__(self, srcpath, selistDf=None,lossweight=None,conopt=None):
        super(traincls_dataset_wmixed_gmorani_pathUp2, self).__init__()
        self.list=[os.path.split(f)[-1] for f in selistDf['tfl'].values]
        self.srcpath=srcpath
        self.lossweight=selistDf[lossweight].values if lossweight is not None else None
        self.conopt=conopt

    def __getitem__(self, index):
        f_path=self.srcpath + '/' +self.list[index]
        #f_path=self.list[index]
        #print('f_path:',f_path)
        assert os.path.exists(f_path)
        try:
            adata=np.load(f_path,allow_pickle=True)
            mask=adata['orgmask']
            mask=mask[None,:,:]
            img=adata['orgimg'].astype(np.float32)/256.0
            img = np.moveaxis(img, -1, 0)
        except OSError:
            return None, None, None,None
        conopt=-1 if self.conopt is None else adata[self.conopt]
        losswei=-1 if self.lossweight is None else self.lossweight[index]
        bflname=Path(f_path).stem
        return img, mask, bflname 

    def __len__(self):
        return len(self.list)      
      

class mask_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0):
        super(mask_dataset, self).__init__()
        self.list = glob(data_path+'/**/*.npz', recursive=True)
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        mask_path = self.list[index]
        assert os.path.exists(mask_path) 
        try:
          adata=np.load(mask_path,allow_pickle=True)
          img=adata['img']
        #  img.dtype='int8'
          mask=adata['mask'] 
          mask=mask[None,:,:] 
        except OSError:
          return None, None, None
       # print('img.dtype, mask.dtype: ',img.shape,';', img.dtype,';',mask.shape,';',mask.dtype)
        return img,mask, os.path.basename(self.list[index])

    def __len__(self):
        return len(self.list)  
      
