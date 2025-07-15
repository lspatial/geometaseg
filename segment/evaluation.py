from __future__ import print_function
import torch
import pandas as pd
from torch.nn import functional as F
import torchmetrics
from segment.metrics import ComputeIoU
import datetime 
from utils import cov2time

def evaluateWei(modeltype,epoch,criterion,acc_t,train_loader,test_loader,model,device,threshold,conopt=None,inputborder=None):
    torch.cuda.empty_cache()
    masks_pred_all,true_masks_all=[],[]
    tlosswei=[]
    compute_iou=ComputeIoU(2)
    print('     retrieve training metrics...... ')
    start =datetime.datetime.now() 
    for tsample_img, mask, conopt, losswei  in train_loader:
        tsample_img =tsample_img.to(device)
        true_masks =mask.to(device)
        if inputborder is not None and isinstance(inputborder, int):
            tsample_img = tsample_img[:, :, inputborder:-inputborder, inputborder:-inputborder]
            true_masks = true_masks[:, :, inputborder:-inputborder, inputborder:-inputborder]
        masks_pred = model(tsample_img)
        if modeltype == 'segformer':
            masks_pred = masks_pred['logits']
            masks_pred = torch.nn.functional.interpolate(
                masks_pred,
                size=tsample_img.shape[2:],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
        masks_pred=masks_pred.cpu().detach()
        true_masks=true_masks.cpu().detach()
        masks_pred_all.append(masks_pred)
        true_masks_all.append(true_masks)
        tlosswei.append(losswei)
        del tsample_img, mask,true_masks, masks_pred
        torch.cuda.empty_cache()
    masks_pred_all = torch.cat(masks_pred_all)
    masks_pred_all=masks_pred_all[:,1] if conopt is not None else masks_pred_all
    true_masks_all = torch.cat(true_masks_all)
    tlosswei = torch.cat(tlosswei)
    masks_pred_all = torch.squeeze(masks_pred_all)
    true_masks_all = torch.squeeze(true_masks_all)
    true_masks_all = true_masks_all[:, 10:-10, 10:-10]
    masks_pred_all = masks_pred_all[:, 10:-10, 10:-10]
    masks_pred_allF=F.sigmoid(masks_pred_all)
    masks_pred_all_cls=(masks_pred_allF>=threshold).float()*1
    compute_iou(masks_pred_all_cls.numpy(),true_masks_all.numpy().astype(int))
    ious=compute_iou.get_ious()
    miou = compute_iou.get_miou()# ignore=0
    tlosswei=tlosswei/torch.sum(tlosswei)
    loss=criterion(masks_pred_all,true_masks_all.float(),tlosswei).numpy()
    acc=acc_t(masks_pred_all,true_masks_all).numpy()
    jaccard_t = torchmetrics.JaccardIndex(task='binary') 
    jtt=jaccard_t(masks_pred_all_cls[:,None,:,:],true_masks_all[:,None,:,:]).numpy()
    end =datetime.datetime.now()
    print('Retrieving took about ',cov2time(start,end)) 
    print('Training metrics(entrmorani): loss:',loss, ',acc:', acc,', jt:',jtt,', miou:',miou,',ious:',ious)
    masks_pred_all,true_masks_all=[],[]
    tlosswei=[]
    print('     test metrics...... ')
    start =datetime.datetime.now() 
    for tsample_img, mask, conopt, losswei  in test_loader:
        tsample_img =tsample_img.to(device)
        true_masks =mask.to(device)
        if inputborder is not None and isinstance(inputborder, int):
            tsample_img = tsample_img[:, :, inputborder:-inputborder, inputborder:-inputborder]
            true_masks = true_masks[:, :, inputborder:-inputborder, inputborder:-inputborder]
        masks_pred = model(tsample_img)
        if modeltype == 'segformer':
            masks_pred = masks_pred['logits']
            masks_pred = torch.nn.functional.interpolate(
                masks_pred,
                size=tsample_img.shape[2:],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
        masks_pred=masks_pred.cpu().detach()
        true_masks=true_masks.cpu().detach()
        masks_pred_all.append(masks_pred)
        true_masks_all.append(true_masks)
        tlosswei.append(losswei)
        del tsample_img, mask, true_masks, masks_pred
        torch.cuda.empty_cache()
    masks_pred_all = torch.cat(masks_pred_all)
    masks_pred_all=masks_pred_all[:,1] if conopt is not None else masks_pred_all
    true_masks_all = torch.cat(true_masks_all)
    tlosswei = torch.cat(tlosswei)
    masks_pred_all = torch.squeeze(masks_pred_all)
    true_masks_all = torch.squeeze(true_masks_all)
    true_masks_all = true_masks_all[:, 10:-10, 10:-10]
    masks_pred_all = masks_pred_all[:, 10:-10, 10:-10]
    compute_iou=ComputeIoU(2)
    masks_pred_allF=F.sigmoid(masks_pred_all)
    masks_pred_all_cls = (masks_pred_allF >= threshold).float() * 1
    compute_iou(masks_pred_all_cls.numpy().astype(int), true_masks_all.numpy().astype(int))
    tious = compute_iou.get_ious()
    tmiou = compute_iou.get_miou()
    tlosswei=tlosswei/torch.sum(tlosswei)
    tloss=criterion(masks_pred_all,true_masks_all.float(),tlosswei).numpy()
    tacc = acc_t(masks_pred_all, true_masks_all).numpy()
    jaccard_t = torchmetrics.JaccardIndex(task='binary')        
    tjtt = jaccard_t(masks_pred_all_cls[:, None, :, :], true_masks_all[:, None, :, :]).numpy()
    end =datetime.datetime.now()
    print('Testing took about ',cov2time(start,end)) 
    print('Testing metrics(entrmorani): loss:',tloss, ',acc:', tacc,', jt:',tjtt,', miou:',tmiou,',ious:',tious)
    ametrics=pd.DataFrame({'epoch':epoch,'train_loss':loss,'train_acc':acc,'train_jt':jtt,'train_miou':miou,
                  'train_iou_0':ious[0],'train_iou_1':ious[1], 
                  'test_loss':tloss,'test_acc':tacc,'test_jt':tjtt,'test_miou':tmiou,
                  'test_iou_0':tious[0],'test_iou_1':tious[1]},index=[epoch])
    return ametrics 

def evaluate(modeltype,epoch,criterion,acc_t,train_loader,test_loader,model,device,threshold,conopt=None,inputborder=None):
    torch.cuda.empty_cache()
    masks_pred_all,true_masks_all=[],[]
    compute_iou=ComputeIoU(2)
    print('     retrieve training metrics...... ')
    start =datetime.datetime.now()
    for tsample_img, mask, _, _  in train_loader:
        tsample_img =tsample_img.to(device)
        true_masks =mask.to(device)
        if inputborder is not None and isinstance(inputborder, int):
            tsample_img = tsample_img[:, :, inputborder:-inputborder, inputborder:-inputborder]
            true_masks = true_masks[:, :, inputborder:-inputborder, inputborder:-inputborder]
        masks_pred = model(tsample_img)
        if modeltype == 'segformer':
            masks_pred = masks_pred['logits']
            masks_pred = torch.nn.functional.interpolate(
                masks_pred,
                size=tsample_img.shape[2:],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
        masks_pred=masks_pred.cpu().detach()
        true_masks=true_masks.cpu().detach()
        masks_pred_all.append(masks_pred)
        true_masks_all.append(true_masks)
        del tsample_img, mask,true_masks, masks_pred
        torch.cuda.empty_cache()
    masks_pred_all = torch.cat(masks_pred_all)
    masks_pred_all=masks_pred_all[:,1] if conopt is not None else masks_pred_all
    true_masks_all = torch.cat(true_masks_all)
    masks_pred_all=torch.squeeze(masks_pred_all)
    true_masks_all=torch.squeeze(true_masks_all)
    true_masks_all=true_masks_all[:,10:-10,10:-10]
    masks_pred_all=masks_pred_all[:,10:-10,10:-10]
    masks_pred_allF=F.sigmoid(masks_pred_all)
    masks_pred_all_cls=(masks_pred_allF>=threshold).float()*1
    compute_iou(masks_pred_all_cls.numpy(),true_masks_all.numpy().astype(int))
    ious=compute_iou.get_ious()
    miou = compute_iou.get_miou()# ignore=0
    loss=criterion(masks_pred_all,true_masks_all.float()).numpy()
    acc=acc_t(masks_pred_all,true_masks_all).numpy()
    jaccard_t = torchmetrics.JaccardIndex(task='binary')
    jtt=jaccard_t(masks_pred_all_cls[:,None,:,:],true_masks_all[:,None,:,:]).numpy()
    end =datetime.datetime.now()
    print('Retrieving took about ',cov2time(start,end))
    print('Training metrics(entrmorani): loss:',loss, ',acc:', acc,', jt:',jtt,', miou:',miou,',ious:',ious)
    masks_pred_all,true_masks_all=[],[]
    print('     test metrics...... ')
    start =datetime.datetime.now()
    for tsample_img, mask, _, _  in test_loader:
        tsample_img =tsample_img.to(device)
        true_masks =mask.to(device)
        if inputborder is not None and isinstance(inputborder, int):
            tsample_img = tsample_img[:, :, inputborder:-inputborder, inputborder:-inputborder]
            true_masks = true_masks[:, :, inputborder:-inputborder, inputborder:-inputborder]
        masks_pred = model(tsample_img)
        if modeltype == 'segformer':
            masks_pred = masks_pred['logits']
            masks_pred = torch.nn.functional.interpolate(
                masks_pred,
                size=tsample_img.shape[2:],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
        masks_pred=masks_pred.cpu().detach()
        true_masks=true_masks.cpu().detach()
        masks_pred_all.append(masks_pred)
        true_masks_all.append(true_masks)
        del tsample_img, mask, true_masks, masks_pred
        torch.cuda.empty_cache()
    masks_pred_all = torch.cat(masks_pred_all)
    masks_pred_all = masks_pred_all[:, 1] if conopt is not None else masks_pred_all
    true_masks_all = torch.cat(true_masks_all)
    masks_pred_all = torch.squeeze(masks_pred_all)
    true_masks_all = torch.squeeze(true_masks_all)
    true_masks_all=true_masks_all[:,10:-10,10:-10]
    masks_pred_all=masks_pred_all[:,10:-10,10:-10]
    compute_iou=ComputeIoU(2)
    masks_pred_allF=F.sigmoid(masks_pred_all)
    masks_pred_all_cls = (masks_pred_allF >= threshold).float() * 1
    compute_iou(masks_pred_all_cls.numpy().astype(int), true_masks_all.numpy().astype(int))
    tious = compute_iou.get_ious()
    tmiou = compute_iou.get_miou()
    tloss=criterion(masks_pred_all,true_masks_all.float()).numpy()
    tacc = acc_t(masks_pred_all, true_masks_all).numpy()
    jaccard_t = torchmetrics.JaccardIndex(task='binary')
    tjtt = jaccard_t(masks_pred_all_cls[:, None, :, :], true_masks_all[:, None, :, :]).numpy()
    end =datetime.datetime.now()
    print('Testing took about ',cov2time(start,end))
    print('Testing metrics(entrmorani): loss:',tloss, ',acc:', tacc,', jt:',tjtt,', miou:',tmiou,',ious:',tious)
    ametrics=pd.DataFrame({'epoch':epoch,'train_loss':loss,'train_acc':acc,'train_jt':jtt,'train_miou':miou,
                  'train_iou_0':ious[0],'train_iou_1':ious[1],
                  'test_loss':tloss,'test_acc':tacc,'test_jt':tjtt,'test_miou':tmiou,
                  'test_iou_0':tious[0],'test_iou_1':tious[1]},index=[epoch])
    return ametrics