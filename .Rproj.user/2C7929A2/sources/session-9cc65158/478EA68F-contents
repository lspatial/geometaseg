from __future__ import print_function
import torch
import pandas as pd
from torch.nn import functional as F
import torchmetrics
from segment.metrics import ComputeIoU
import datetime 
from utils import cov2time
import torchmetrics 
import numpy as np 

def retrieveRes(imgs,gtrues,threshold,pmasks,rpath,bnames,flag):
    if pmasks.shape[0]>1:
      pmasks=torch.squeeze(pmasks)
    else:
      pmasks=pmasks[:,0,:,:]
    if gtrues.shape[0]>1:
      gtrues=torch.squeeze(gtrues)
    else:
      gtrues=gtrues[:,0,:,:]
    imgs=imgs[:,:,10:-10,10:-10]
    gtrues=gtrues[:,10:-10,10:-10]
    pmasks=pmasks[:,10:-10,10:-10]
    pmasks=F.sigmoid(pmasks)
    pmasks=(pmasks>=threshold).float()*1
    compute_iou=ComputeIoU(2)
    acc_t = torchmetrics.Accuracy(task='binary')
    for i in range(imgs.shape[0]):
      aimg=imgs[i]
      apmask=pmasks[i]
      atrue=gtrues[i]
      compute_iou(apmask.numpy(),atrue.numpy().astype(int))
      ious=compute_iou.get_ious()
      miou = compute_iou.get_miou()# ignore=0
      acc=acc_t(apmask,atrue).numpy()
      aimgDf=pd.DataFrame({'miou':miou,'ious_0':ious[0],'ious_1':ious[1],'acc':acc},index=[flag])
      mfile=rpath+'/'+bnames[i]+'_'+flag+'_'+str(i)+'_iou1_'+str(round(ious[1], 2))+'.npz'
      np.savez(mfile,img=aimg, truemask=atrue,premask=apmask)
      mfile=rpath+'/'+bnames[i]+'_'+flag+'_'+str(i)+'_iou1_'+str(round(ious[1],2))+'_metrics.csv'
      aimgDf.to_csv(mfile,index=False)



def predict(modeltype,data_loader,model,device,threshold,inputborder=None,rpath='/tmp',rflag='tmp'):
    torch.cuda.empty_cache()
    imgs_all,masks_pred_all,true_masks_all=[],[],[]
    compute_iou=ComputeIoU(2)
    counter=1
    for tsample_img, mask, basenames in data_loader:
        if mask.shape[0]==1:
          continue 
        tsample_img =tsample_img.to(device)
        true_masks =mask.to(device)
        if inputborder is not None and isinstance(inputborder, int):
            tsample_img = tsample_img[:, :, inputborder:-inputborder, inputborder:-inputborder]
            true_masks = true_masks[:, :, inputborder:-inputborder, inputborder:-inputborder]
        masks_pred = model(tsample_img)
        if modeltype.startswith('segformer'):
            masks_pred = masks_pred['logits']
            masks_pred = torch.nn.functional.interpolate(
                masks_pred,
                size=tsample_img.shape[2:],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
        masks_pred=masks_pred.cpu().detach()
        true_masks=true_masks.cpu().detach()
        tsample_img=tsample_img*256
        tsample_img=tsample_img.int().cpu().detach()
        retrieveRes(tsample_img,true_masks,threshold,masks_pred,rpath,basenames,flag=rflag)
        del tsample_img, mask,true_masks, masks_pred
        torch.cuda.empty_cache()
        counter+=1
