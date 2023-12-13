#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    samplegid5.py [options]

Options:
    -h --help                               show this screen.
    --gpu=<int>                             use indexed GPU, -1 means no GPU [default: 0]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 12]
    --train-prp=<float>                     train sample proportion [default: 0.1]
    --test-prp=<float>                      test sample proportion [default: 0.2]
    --testsample-type=<str>                 test sample type [default: random]
    --n-epoch=<int>                         epoch number [default: 2]
    --stratify=<str>                        stratification factor [default:None]
    --stratify-rng=<int or list>            stratifum number [default: 5]
    --weight=<str>                          weight factor [default:None]
    --model-type=<str>                      model type [default: unet]
    --loss-weight=<str>                     weight field for loss function [default:None]
    --con-opt=<str>                         constrained optimization [default:None]
    --con-opt-w=<tuple<float>>              weights for BCE and constrained optimization [default:None]
    --target-class=<str>                    classification [default: buildup]
    --log-dir=<str>                         path to save the results [default: /geosampling/rs_gidcls5/meta_swin_log]
    --pretrain-mode=<str>                   pretrain mode [default: all]
"""

from __future__ import print_function
import os

import pandas as pd
import torch
from datagid5.dataload import traincls_dataset_wmixed_gmorani_pathUp
import numpy as np
from sampling.sampling import sampling
from model.gunet import UNet
from segment.train import train 
from docopt import docopt 
import re
from model.swinunetlp import SwinUnetLP
from transformers import SegformerForSemanticSegmentation
from model.metamodel import MetaUnet

class samplegid5:
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

  def retrieveModel(self,modeltype,modelconfig):
     if modeltype=='unet':
        model=UNet(**modelconfig)
        return model
     elif modeltype=='swinunet':
         config={'model_pretrain_ckpt':'/aiprof/pretrained/swin_tiny_patch4_window7_224.pth',
                 '--img-size':224,'--swin-patch-size':4, '--model-name':'swin_tiny_patch4_window7_224',
                 '--model-swin-embed-dim':96,'--model-swin-in-chans':3,'--model-swin-depths':[2, 2, 6, 2],
                 '--model-swin-num-heads':[3, 6, 12, 24],'--model-swin-window-size':7,'--model-drop-rate':0,
                 '--model-swin-mlp-ratio':4,'--model-swin-qkv-bias':True,'--model-swin-qk-scale':None,
                 '--model-drop-path-rate':0.1,'--model-swin-ape':False,'--model-swin-patch-norm':True,
                 '--train-use-checkpoint':False}
         model=SwinUnetLP(config,num_classes=1)
         model.load_from(config)
         return model
     elif modeltype=='segformer':
         pretrained_model_name = "nvidia/mit-b3"
         id2label = {0: "buildup"}
         label2id = {label: id for id, label in id2label.items()}
         num_labels = len(id2label)
         model = SegformerForSemanticSegmentation.from_pretrained(
             pretrained_model_name,
             num_labels=num_labels,
             id2label=id2label,
             label2id=label2id,
         )
         return model
     elif modeltype=='meta_train':
         model=MetaUnet()
         return model 
     elif modeltype=='meta_train_mcls':
         tfl='/geosampling/metasegement/test8/state-1.pt'
         model=MetaUnet(pretainedfl=tfl)
         return model    
     return None
  
  def setPath(self,srcpath,targetpath):
     self.targetpath=targetpath
     self.srcpath=srcpath 

  def runtrain(self,samconfig,modeltype,modelconfig,device,conopt,conopt_w,n_epoch,batch_size,lossweight=None,pretrain_mode='all'):
     trainsampleDf,testsampleDf = self.datasampling(samconfig)
     model=self.retrieveModel(modeltype,modelconfig)
     if not os.path.exists(self.targetpath): 
        os.makedirs(self.targetpath)  
     train_datatset_ = traincls_dataset_wmixed_gmorani_pathUp(self.srcpath,trainsampleDf,lossweight,conopt)
     train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=batch_size, shuffle=True,
                                               num_workers=2)
    
     test_datatset_ = traincls_dataset_wmixed_gmorani_pathUp(self.srcpath,testsampleDf,lossweight,None)
     test_loader = torch.utils.data.DataLoader(dataset=test_datatset_, batch_size=batch_size, shuffle=False,
                                              num_workers=2)
     inputborder=26 if modeltype=='swinunet' else (10 if modeltype=='unet' else None)
     train(modeltype,model,device,self.targetpath,train_loader,test_loader,lossweight,conopt,conopt_w,\
                   show_time=6,npoch=n_epoch,inputborder=inputborder,pretrainmode=pretrain_mode)


def main():
    args = docopt(__doc__)
    print(args)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    samtest=samplegid5()
    targetcls=args['--target-class']
    trainp=float(args['--train-prp'])
    testp=float(args['--test-prp'])
    conopt=args['--con-opt']
    gpu=int(args['--gpu'])
    n_epoch=int(args['--n-epoch'])
    loss_weight=args['--loss-weight']
    batch_size=int(args['--batch-size'])
    log_dir=args['--log-dir']
    pretrain_mode=args['--pretrain-mode']
    if args['--stratify-rng'].__contains__('[') and args['--stratify-rng'].__contains__(']'):
        strange=re.split(',', re.sub('\[|\]', '', args['--stratify-rng']))
        strange=[float(n) for n in strange]
    elif args['--stratify-rng'].isnumeric():
        strange= int(args['--stratify-rng'])
    else:
        strange=None
    if args['--con-opt-w'] is not None and args['--con-opt-w'].__contains__('(') and args['--con-opt-w'].__contains__(')'):
        conopt_w=re.split(',', re.sub('\(|\)', '', args['--con-opt-w']))
        conopt_w=tuple([float(n) for n in conopt_w])
    else:
        conopt_w=(0.5,0.5)
    if gpu==-1:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:'+str(gpu))
    clsmap={'background':0,'water':1,'buildup':2,'farmland':3,'forest':4,'meadow':5}
    cls=clsmap[targetcls]
    srcpath='/geosampling/rs_gidcls5/cdist10_r256sc1_mcomplex/'+str(cls)
    targetpath =log_dir+'/SwinUNet_Model_test1_'+str(trainp)
    if  not os.path.exists(targetpath): 
        os.makedirs(targetpath)
    configDf = pd.DataFrame.from_dict([args])
    configDf.to_csv(targetpath+"/config.csv", header=False, index=True)
    samtest.setPath(srcpath,targetpath)
    samconfig={'sampleSFile':'/geosampling/rs_gidcls5/cdist10_r256sc1_mcomplex_sum/patchall_cls2_sum.csv',\
               'uid':'uid','trainprp':trainp,'testprp':testp,'testsamplestype':args['--testsample-type'],\
               'stratify':args['--stratify'],'strange':strange,'weight':args['--weight']}
    modeltype=args['--model-type']# 'unet' # n_channels, n_classes,encoders, bilinear=False
    noutput=2 if conopt is not None else 1
    modelconfig={'n_channels':3, 'n_classes':noutput ,'encoders':[16,32,64,128,256,512,1024,2048],'bilinear':True}
    samtest.runtrain(samconfig,modeltype,modelconfig,device,conopt,conopt_w,n_epoch,batch_size,loss_weight,pretrain_mode)

if __name__=='__main__':
    main()
