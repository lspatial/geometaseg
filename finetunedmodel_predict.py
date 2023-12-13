#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os

import pandas as pd
import torch
from datagid5.dataload import traincls_dataset_wmixed_gmorani_pathUp2
import numpy as np
from sampling.sampling import sampling
from model.gunet import UNet
from segment.predict import predict 
from docopt import docopt 
import re
from model.swinunetlp import SwinUnetLP
from transformers import SegformerForSemanticSegmentation
from model.metamodel import MetaUnet

class samplegid5predict:
  def __init__(self):
     self.__version__='0.0.1'

  def datasampling(self,samconf):
     sampleSFile=samconf['sampleSFile']
     uid=samconf['uid']
     trainprp=samconf['trainprp']
     testprp=samconf['testprp']
     testsamplestype= samconf.get('testsamplestype','random')
     stratify= samconf.get('stratify',None)
     strange=samconf.get('strange',None)
     weight=samconf.get('weight',None)
     trainsampleDf,testsampleDf = sampling(sampleSFile,uid,trainprp,\
                 testprp, testsamplestype, stratify,strange, weight)
     return trainsampleDf,testsampleDf 

  def retrieveModel(self,modeltype,device):
     if modeltype=='unet':
        tainedfl='/geosampling/pretrained_3/unet_full/SwinUNet_Model_test1_0.1/model_unet.tor'
        model=torch.load(tainedfl,map_location=device)
        return model
     elif modeltype=='segformer_full':
        tainedfl='/aiprof/segform_pretrained/seg_finetune_all_2_2/model_segformer.tor'
        model=torch.load(tainedfl,map_location=device)
        return model
     elif modeltype=='segformer_decoder':
        tainedfl='/aiprof/segform_pretrained/seg_finetune_decoder_1_4/best_model_segformer.tor'
        model=torch.load(tainedfl,map_location=device)
        return model
     elif modeltype=='meta_train':
        tainedfl='/geosampling/pretrained_2/meta_all_0.2/SwinUNet_Model_test1_0.2/best_model_meta_train.tor'
        model=torch.load(tainedfl,map_location=device)
        return model
     return None
  
  def setPath(self,srcpath,targetpath):
     self.targetpath=targetpath
     self.srcpath=srcpath 

  def runpredict(self,samconfig,device,batch_size,lossweight=None):
     trainsampleDf,testsampleDf = self.datasampling(samconfig)
     if not os.path.exists(self.targetpath): 
        os.makedirs(self.targetpath)  
     predict_datatset_ = traincls_dataset_wmixed_gmorani_pathUp2(self.srcpath,trainsampleDf,lossweight,None)
     predict_loader = torch.utils.data.DataLoader(dataset=predict_datatset_, batch_size=batch_size, shuffle=False,
                                              num_workers=2)
     models=['unet','segformer_full','segformer_decoder','meta_train'] 
     for modeltype in models:
        model=self.retrieveModel(modeltype,device)
        inputborder=26 if modeltype=='swinunet' else (10 if modeltype=='unet' else None)
        predict(modeltype,predict_loader,model,device,0.5,inputborder,self.targetpath,rflag='m_'+modeltype+'_')


def main():
    samtest=samplegid5predict()
    gpu=1
    if gpu==-1:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:'+str(gpu))
    clsmap={'background':0,'water':1,'buildup':2,'farmland':3,'forest':4,'meadow':5}
    cls=clsmap['buildup']
    srcpath='/geosampling/rs_gidcls5/cdist10_r256sc1_mcomplex/'+str(cls)
    targetpath ='/geosampling/predict4'
    if  not os.path.exists(targetpath): 
        os.makedirs(targetpath)
    samtest.setPath(srcpath,targetpath)
    samconfig={'sampleSFile':'/geosampling/rs_gidcls5/cdist10_r256sc1_mcomplex_sum/patchall_cls2_sum.csv',\
               'uid':'uid','trainprp':0.0001,'testprp':0.0001,'testsamplestype':'random',\
               'stratify':None,'strange':None,'weight':'p_all'}
    batch_size=36                
    samtest.runpredict(samconfig,device,batch_size)

if __name__=='__main__':
    main()


