#from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#import torchvision.utils as vutils
from torch.autograd import Variable
import time
import numpy as np
from numpy import *
from datagid5.dataload import mask_dataset 
from gcomplex.com2cov import Comp2dConv 
import re 
from PIL import Image
import shutil 


def asrc2complexity(maskpath,outpath,maxsize=21):
  batch_size=120
  maxksize=21 
  mkernel_sz=[21,15,11,7,5]
  margin_border_size=maxksize //2
  mask_dataset_ = mask_dataset(maskpath) 
  mask_loader = torch.utils.data.DataLoader(dataset=mask_dataset_, 
              batch_size=batch_size, shuffle=True,num_workers=60)
  #dataiter = iter(mask_loader)
   
  for imgs, masks, names  in  mask_loader:  #range(0, mask_dataset_.__len__(), batch_size):
      #imgs, masks, names = dataiter.next()
      if imgs is None or masks is None: 
          continue
      print(imgs.shape,masks.shape)
      bn,wd,hei,ch =imgs.shape
      #print('names:',names)
      dataDic={}
      
      for kernel_size in mkernel_sz:
          print('kernel_size:',kernel_size) 
          border_size = kernel_size // 2 
          border_dif = margin_border_size-border_size 
          complexnet = Comp2dConv(1,1,kernel_size=kernel_size, stride=1, padding=0,
                        ctype='percent',device=None,defmean=None) 
          iscuda=True
          num_GPU=1 
          if iscuda:
              complexnet.cuda()
          if num_GPU > 1:
              complexnet = nn.DataParallel(complexnet)   
          complexity = complexnet(masks[:,:,border_dif:(wd-border_dif),border_dif:(hei-border_dif)])
          complexity_np=complexity.cpu().numpy().astype(np.float32)
          dataDic[str(kernel_size)+'_per']=complexity_np      
      
      for kernel_size in mkernel_sz:
          print('kernel_size:',kernel_size) 
          border_size = kernel_size // 2 
          border_dif = margin_border_size-border_size 
          complexnet = Comp2dConv(1,1,kernel_size=kernel_size, stride=1, padding=0,
                        ctype='entropy',device=None,defmean=None) 
          iscuda=True
          num_GPU=1 
          if iscuda:
              complexnet.cuda()
          if num_GPU > 1:
              complexnet = nn.DataParallel(complexnet)   
          complexity = complexnet(masks[:,:,border_dif:(wd-border_dif),border_dif:(hei-border_dif)])
          complexity_np=complexity.cpu().numpy().astype(np.float32)
          dataDic[str(kernel_size)+'_en']=complexity_np      
      
          
      for kernel_size in mkernel_sz:
          print('kernel_size:',kernel_size) 
          border_size = kernel_size // 2 
          border_dif = margin_border_size-border_size 
          complexnet = Comp2dConv(1,1,kernel_size=kernel_size, stride=1, padding=0,
                        ctype='gmorani',morantype='rock',device=None,defmean=None) 
          iscuda=True
          num_GPU=1 
          if iscuda:
              complexnet.cuda()
          if num_GPU > 1:
              complexnet = nn.DataParallel(complexnet)   
          complexity = complexnet(masks[:,:,border_dif:(wd-border_dif),border_dif:(hei-border_dif)])
          complexity_np=complexity.cpu().numpy().astype(np.float32)
          dataDic[str(kernel_size)+'_moran']=complexity_np 
          
      
      for k in range(imgs.shape[0]):
          fname=re.sub('\\.npz', '_imgcomask.npz',names[k])
          mfile=outpath+'/'+fname
          orgimg=imgs[k,:,:,:]
          inimg=imgs[k,margin_border_size:(wd-margin_border_size),margin_border_size:(hei-margin_border_size),:]
          orgmask=masks[k,0,:,:]
          inmask=masks[k,0,margin_border_size:(wd-margin_border_size),margin_border_size:(hei-margin_border_size)]
          cmdStr='np.savez("'+mfile+'", orgimg=orgimg,orgmask=orgmask,img=inimg,mask=inmask'
          for kernel_size in mkernel_sz: 
            cmdStr=cmdStr+',m'+str(kernel_size)+'=dataDic["'+str(kernel_size)+'_moran"]['+str(k)+',:,:]'
            cmdStr=cmdStr+',e'+str(kernel_size)+'=dataDic["'+str(kernel_size)+'_en"]['+str(k)+',:,:]'
            cmdStr=cmdStr+',p'+str(kernel_size)+'=dataDic["'+str(kernel_size)+'_per"]['+str(k)+',:,:]'
          cmdStr=cmdStr+')'
          eval(cmdStr)
        
        
clsssDict1={0:'background',1:'water',2:'buildup',3:'farmland',4:'forest', 5:'meadow'}

for k,v in clsssDict1.items():
  if k!=0:
    print(k,v,' ... ...')
    maskpath='/geosampling/complexRSPatches/gid5_10/'+str(k) 
    outpath='/geosampling/complexRSPatches/gid5_10_mcomplex/'+str(k) 
    if os.path.exists(outpath):
      shutil.rmtree(outpath)
    os.makedirs(outpath) 
    asrc2complexity(maskpath,outpath)


                                
                                
                                
                                
                                
                                
                                
