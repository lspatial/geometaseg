
from __future__ import print_function
import torch
import numpy as np
import sys
import pickle 


def anaCmplex(f_path,tarcom=[11,15,21,7]):
  adata=np.load(f_path,allow_pickle=True)
  #['orgimg', 'img', 'mask', 'k21', 'k15', 'k11', 'k7', 'k5']
  compSum=0
  for acom in tarcom:
    akey='k'+str(acom)
    acompV=adata[akey]
    compSum=compSum+acompV.sum()
  wei=compSum/(acompV.shape[0]*acompV.shape[1]*len(tarcom)*1.0)
  return wei 

def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    labels = torch.tensor(batch[0], dtype=torch.int32)
    imgs = batch[1]
    del batch
    return labels, imgs

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def cov2time(start,end):
  duration = end - start
  hour = duration.seconds//3600
  minute = duration.seconds//60
  second = duration.seconds % 60
  res=str(hour)+':'+ str(minute)+':'+str(second)
  return res 

def is_float(text):
    try:
        float(text)
        # check for nan/infinity etc.
        if text.isalpha():
            return False
        return True
    except ValueError:
        return False

def retrieveArgs():
  gpu=0 
  icls=1
  if len(sys.argv)==1:
    trainp=0.6
    testp=0.3
  elif len(sys.argv)==2:
    _,trainp=sys.argv
    trainp=float(trainp)
    testp=0.3 
  elif len(sys.argv)==3:
    _,trainp, testp=sys.argv
    trainp=float(trainp)
    testp=float(testp)
  elif len(sys.argv)==4:
    _,trainp, testp, gpu=sys.argv
    trainp=float(trainp)
    testp=float(testp)
    gpu=int(gpu)
    icls=1
  elif len(sys.argv)==5:
    _,trainp, testp, gpu, icls=sys.argv
    trainp=float(trainp)
    testp=float(testp)
    gpu=int(gpu)
    icls=int(icls)
  return trainp, testp, gpu,icls 
