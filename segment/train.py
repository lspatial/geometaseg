from __future__ import print_function
from typing import List, Tuple, Iterable
from segment.parameters_to_fine_tune import parameters_to_fine_tune,LoRALayerWrapper
import torch
import time
import pandas as pd
from segment.lossmetrics import DiceBCELoss,weiMSELoss,DiceBCELossWei,weiMSELossWei
import torchmetrics
import datetime
from segment.evaluation import evaluate,evaluateWei
from utils import cov2time
from transformers import SegformerForSemanticSegmentation

def train(modeltype,model,device,targetpath,train_loader,test_loader,losswei=None,conopt=None,conopt_w=None,
          show_time=6,npoch=100,threshold = torch.tensor([0.5]),inputborder=None,pretrainmode=None):
    model = model.to(device)
    if losswei is not None:
        criterion = DiceBCELossWei()
    else:
        criterion = DiceBCELoss()
    criterion_gpu =criterion.to(device)
    if conopt is not None:
        if losswei is not None:
            criterioncon=weiMSELossWei()
        else:
            criterioncon = weiMSELoss()
        criterioncon_gpu = criterioncon.to(device)
    if pretrainmode=='lora':
        lrank=4
        for m in model.transformer.h:
            m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, lrank)
            m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, lrank)
            m.attn.c_attn = LoRALayerWrapper(m.attn.c_attn, lrank)
    parameters_to_fn: List[torch.nn.Parameter] = None
    if modeltype=='unet' or pretrainmode is None:
        parameters_to_fn=model.parameters()
    elif pretrainmode=='segformer_decoder':
        parameters_to_fn = parameters_to_fine_tune(model, pretrainmode)
    elif modeltype=='swinunet':
        parameters_to_fn=parameters_to_fine_tune(model,pretrainmode)
    else:
        parameters_to_fn = model.parameters()
    optimizer = torch.optim.Adam(parameters_to_fn, lr=0.004, betas=(0.0002, 0.0095))
    start = time.time() 
    model.train() 
    hist = None 
    acc_t = torchmetrics.Accuracy(task='binary')
    dice=torchmetrics.Dice(average='micro')
    best_test_iou_1=-999
    for epoch in range(1, npoch +1):
        print('Epoch ',epoch, ' ... ...')
        start =datetime.datetime.now() 
        for  tsample_img, true_masks,conopts,weights  in train_loader:
            tsample_img =tsample_img.to(device)
            true_masks =true_masks.to(device)
            if inputborder is not None and isinstance(inputborder,int):
                tsample_img=tsample_img[:,:,inputborder:-inputborder,inputborder:-inputborder]
                true_masks=true_masks[:,:,inputborder:-inputborder,inputborder:-inputborder]
            masks_pred = model(tsample_img)
            if modeltype=='segformer':
                masks_pred=masks_pred['logits']
                masks_pred = torch.nn.functional.interpolate(
                    masks_pred,
                    size=tsample_img.shape[2:],  # (height, width)
                    mode='bilinear',
                    align_corners=False
                )
            if conopt is not None:
               conoptV=masks_pred[:,0]
               masks_pred = masks_pred[:,1]
               conopts=conopts.to(device)
            weights=weights/torch.sum(weights)
            weights=weights.to(device)
            loss = criterion_gpu(masks_pred, true_masks.float(),weights)
            if conopt is not None:
                lossopt=criterioncon_gpu(conoptV, conopts.float(),weights)
                loss = loss*conopt_w[0]+lossopt*conopt_w[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del tsample_img,true_masks,masks_pred
            torch.cuda.empty_cache()
        end =datetime.datetime.now()
        print('Training took about ',cov2time(start,end),'loss:',loss.cpu().detach().numpy())
        if epoch==1 or (epoch %  show_time)==0:
           if losswei is not None:
               ametrics = evaluateWei(modeltype,epoch,criterion_gpu,acc_t, train_loader,test_loader,model,device,threshold,conopt,inputborder=inputborder)
           else:
               ametrics = evaluate(modeltype,epoch, criterion_gpu, acc_t, train_loader, test_loader, model, device, threshold,conopt,inputborder=inputborder)
           if hist is None:
              hist = ametrics
           else:
              hist = pd.concat([hist,ametrics],axis=0,ignore_index=True)
        if best_test_iou_1<ametrics['test_iou_1'].values:
           best_test_iou_1=ametrics['test_iou_1'].values
           torch.save(model, targetpath + '/best_model_'+modeltype+'.tor')
           ametrics.to_csv(targetpath + '/best_metrics_'+modeltype+'.csv',index=False)

    torch.save(model.state_dict(), targetpath + '/mstate_'+modeltype+'.pth')
    modelFl = targetpath + '/model_'+modeltype+'.tor'
    torch.save(model, modelFl)
    hist.to_csv(targetpath + '/hist_'+modeltype+'.csv', index=False)
    end = time.time()

    
    
    
