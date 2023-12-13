#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    memaseg.py --learn-inner-lrs --gpu 0 --batch-size 4 --num-group 1 --num-support 10  --num-query 20  --num-inner-steps 1   --init-inner-lr 0.4 --outer-lr 0.03 --num-train-iterations 2 --iloop-batch 10  --oloop-batch 10  --log-dir /geosampling/metasegement/test1  [options]

Options:
    -h --help                               show this screen.
    --batch-size=<int>                      Preset batchsize [default: 6]
    --learn-inner-lrs                       whether to learn the inner learning rate
    --gpu=<int>                             use indexed GPU [default: 0]
    --seed=<int>                            seed [default: 0]
    --num-group=<int>                       number of way (groups) [default: 1]
    --num-support=<int>                     number of support samples [default: 4]
    --num-query=<int>                       number of query samples [default: 4]
    --num-workers=<int>                     worker number [default: 3]
    --num-inner-steps=<int>                 number of inner iterations [default: 1]
    --init-inner-lr=<float>                 initial inner learning rate [default: 0.2]
    --outer-lr=<float>                      outer learning rate [default: 0.01]
    --log-interval=<int>                    interval number [default: 2]
    --val-interval=<int>                    val internal number [default: 2]
    --save-interval=<int>                   save interval number [default: 100]
    --num-test-tasks=<int>                  number of test tasks [default: 50]
    --num-train-iterations=<int>            iteration number [default: 1000]
    --checkpoint-step=<int>                 checkpoint step [default: 10]
    --iloop-batch=<int>                     innerloop batch [default: -1]
    --oloop-batch=<int>                     outer loop batch [default: -1]
    --num-outer-steps=<int>                 outer loop step [default: 1]
    --checkpoint-step=<int>                 checkpoint step [default: -1]
    --log-dir=<str>                         path to save the results [default: /geosampling/metaup/meta_moran_log]
"""
#
#Run example: 
# python3.8 memaseg_mup.py --learn-inner-lrs --gpu 0 --batch-size 4 --num-group 1 --num-support 20  --num-query 20  --num-inner-steps 1   --init-inner-lr 0.4 --outer-lr 0.03 --num-train-iterations 2 --iloop-batch 10  --oloop-batch 10  --log-dir /geosampling/metasegement/test1 
## python3.8 memaseg_mup.py --learn-inner-lrs --gpu 0 --batch-size 4 --num-group 1 --num-support 20  --num-query 20  --num-inner-steps 1   --init-inner-lr 0.4 --outer-lr 0.03 --num-train-iterations 2 --iloop-batch 10  --oloop-batch 10  --log-dir /geosampling/metasegement/test2

from torch.nn import CrossEntropyLoss
from torch import autograd
from model.gunet import UNet
from collections import OrderedDict
import os
import copy
from segment.lossmetrics import DiceBCELoss
from segment.metrics import ComputeIoU
import torchmetrics
import numpy as np
from datagid5.metadataload import get_gid5_dataloader
import torch, gc
from torch.utils import tensorboard
from docopt import docopt
import pandas as pd

LOG_INTERVAL =1
VAL_INTERVAL = LOG_INTERVAL * 2
SAVE_INTERVAL = 100
NUM_TEST_TASKS = 50

class MAMLSeg:
    def __init__(self,basenet,device,num_inner_steps=2,init_inner_lr=0.1,num_outer_steps=2,\
                 learn_inner_lrs=True,outer_lr=0.1,log_dir='/tmp',shreshold=0.5,iloop_batch=5,oloop_batch=5):
        self.metanet=basenet.to(device)
        self._meta_parameters={key: val for key, val in self.metanet.state_dict().items()};
        self._num_inner_steps = num_inner_steps
        self._num_outer_steps = num_outer_steps
        self._inner_lrs = {
            k: torch.tensor(init_inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._start_train_step = 0
        self.device=device
        criterion = CrossEntropyLoss()
        self.criterion=criterion.to(device)
        criterion = CrossEntropyLoss(reduction='sum')
        self.metacriterion = criterion.to(device)
        self.acc_t = torchmetrics.Accuracy(task='binary')
        self.compute_iou = ComputeIoU(2,1)
        self.shrehold=shreshold
        self.iloop_batch=iloop_batch
        self.oloop_batch = oloop_batch

    def clearcudaDeepMemory(self,*targetargs):
        for t in targetargs:
           if torch.is_tensor(t):
               t.detach()
               t.grad = None
               t.storage().resize_(0)
           elif t is torch.nn.Module:
               t.cpu()
           else:
               next
           del t
           gc.collect()
           torch.cuda.empty_cache()

    def _inner_loop(self, images, labels, train):
        #self.taskmodel.load_state_dict(self._meta_parameters)
        #self.taskmodel.load_state_dict(self._meta_parameters)
        taskmodel=copy.deepcopy(self.metanet)
        mious = []
        accs = []
        if self.iloop_batch==-1:
            self.iloop_batch=images.shape[0]
        with torch.enable_grad():
            taskmodel.train()
            #Retrieve initial metrics before adaptation
            masks_pred_all, true_masks_all = [], []
            for batch in range(0, images.shape[0], self.iloop_batch):
                iend = batch + self.iloop_batch
                iend = iend if iend < images.shape[0] else images.shape[0]
                ibatch_imgs = images[batch:iend]
                ibatch_labels = labels[batch:iend]
                ibatch_imgs = ibatch_imgs.to(self.device)
                masks_pred = taskmodel(ibatch_imgs)
                masks_pred = masks_pred.cpu().detach()
                masks_pred_all.append(masks_pred)
                true_masks_all.append(ibatch_labels)
                del ibatch_imgs
                torch.cuda.empty_cache()
            masks_pred_all = torch.cat(masks_pred_all)
            true_masks_all = torch.cat(true_masks_all)
            masks_pred_allF = torch.softmax(masks_pred_all, dim=1)
            masks_pred_all_cls = torch.argmax(masks_pred_allF, axis=1, keepdim=True)
            masks_pred_all_cls = torch.squeeze(masks_pred_all_cls)
            masks_pred_all_cls = torch.stack([1 - masks_pred_all_cls, masks_pred_all_cls], dim=1)
            masks_pred_all_cls = masks_pred_all_cls[:, :, 10:-10, 10:-10]
            labelsdim = torch.squeeze(true_masks_all[:, :, 10:-10, 10:-10])
            self.compute_iou(masks_pred_all_cls[:, 0].cpu().detach().numpy().astype(int),
                             labelsdim[:, 0].cpu().detach().numpy().astype(int))
            miou = self.compute_iou.get_miou()
            acc = self.acc_t(masks_pred_all_cls.cpu().detach(), labelsdim.cpu().detach())
            mious.append(miou)
            accs.append(acc)
            for i in range(0, self._num_inner_steps):
                shindex=torch.randperm(images.shape[0])
                images,labels=images[shindex],labels[shindex]
                for batch in range(0, images.shape[0], self.iloop_batch):
                    iend=batch+self.iloop_batch
                    iend=iend if iend<images.shape[0] else images.shape[0]
                    ibatch_imgs=images[batch:iend]
                    ibatch_labels=labels[batch:iend]
                    ibatch_imgs=ibatch_imgs.to(self.device)
                    ibatch_labels=ibatch_labels.to(self.device)
                    masks_pred = taskmodel(ibatch_imgs)
                    loss = self.criterion(masks_pred, ibatch_labels.float())
                    tparas = taskmodel.state_dict()
                    grads = autograd.grad(loss, taskmodel.parameters(), create_graph=train)
                    with torch.no_grad():
                        for (k, v), g in zip(taskmodel.named_parameters(), grads):
                            tparas[k] = v - self._inner_lrs[k] * g
                    taskmodel.load_state_dict(tparas)
                    del grads,ibatch_imgs,ibatch_labels
                    torch.cuda.empty_cache()
                    grads,ibatch_labels,ibatch_imgs = None,None,None
                    gc.collect()
                masks_pred_all, true_masks_all = [], []
                for batch in range(0, images.shape[0], self.iloop_batch):
                    iend = batch + self.iloop_batch
                    iend = iend if iend < images.shape[0] else images.shape[0]
                    ibatch_imgs = images[batch:iend]
                    true_masks = labels[batch:iend]
                    ibatch_imgs = ibatch_imgs.to(self.device)
                    masks_pred = taskmodel(ibatch_imgs)
                    masks_pred = masks_pred.cpu().detach()
                    masks_pred_all.append(masks_pred)
                    true_masks_all.append(true_masks)
                    del ibatch_imgs,  true_masks, masks_pred
                    torch.cuda.empty_cache()
                masks_pred_all = torch.cat(masks_pred_all)
                true_masks_all = torch.cat(true_masks_all)
                masks_pred_allF = torch.softmax(masks_pred_all,dim=1)
                masks_pred_all_cls = torch.argmax(masks_pred_allF,axis=1,keepdim=True)
                masks_pred_all_cls = torch.squeeze(masks_pred_all_cls)
                masks_pred_all_cls=torch.stack([1-masks_pred_all_cls,masks_pred_all_cls],dim=1)
                masks_pred_all_cls=masks_pred_all_cls[:,:,10:-10,10:-10]
                labelsdim=torch.squeeze(true_masks_all[:,:,10:-10,10:-10])
                self.compute_iou(masks_pred_all_cls[:,0].cpu().detach().numpy().astype(int), labelsdim[:,0].cpu().detach().numpy().astype(int))
                miou = self.compute_iou.get_miou()
                acc = self.acc_t(masks_pred_all_cls.cpu().detach(), labelsdim.cpu().detach())
                mious.append(miou)
                accs.append(acc)
                torch.cuda.synchronize()
            del masks_pred_all_cls, masks_pred_allF,labelsdim,images
            torch.cuda.empty_cache()
            gc.collect()
        return taskmodel,accs,mious

    def _outer_step(self, task_batch, train):
        accuracies_support_batch = []
        mious_support_batch = []
        accuracy_query_meta = []
        miou_query_meta = []
        outer_loss_meta = []
        taskbloss=[]
        taskmodels=[]
        for task in task_batch:
            images_support, labels_support, _, _ = task
            taskmodel,accs,mious = self._inner_loop(images_support, labels_support, train)
            accuracies_support_batch.append(accs)
            mious_support_batch.append(mious)
            taskmodels.append(taskmodel.cpu())
        tnumsample=torch.tensor(images_support.shape[-1]*images_support.shape[-2]*images_support.shape[0],dtype=torch.float)
        for i in range(0, self._num_outer_steps):
            outer_loss_alltasks = []
            for k,task in enumerate(task_batch):
                _, _, images_query, labels_query = task
                if self.oloop_batch == -1:
                    self.oloop_batch = images_query.shape[0]
                shindex=torch.randperm(images_query.shape[0])
                images_query,labels_query=images_query[shindex],labels_query[shindex]
                masks_pred_all, true_masks_all = [], []
                ataskloss = []
                for batch in range(0, images_query.shape[0], self.oloop_batch):
                    iend = batch + self.oloop_batch
                    iend = iend if iend < images_query.shape[0] else images_query.shape[0]
                    ibatch_imgs = images_query[batch:iend]
                    ibatch_labels = labels_query[batch:iend]
                    ibatch_imgs = ibatch_imgs.to(self.device)
                    ibatch_labels = ibatch_labels.to(self.device)
                    ataskmodel=taskmodels[k]
                    ataskmodel=ataskmodel.to(self.device)
                    outputs_query = ataskmodel(ibatch_imgs)
                    loss = self.metacriterion(outputs_query, ibatch_labels.float())
                    ataskloss.append(loss)
                    masks_pred_all.append(outputs_query.cpu().detach())
                    true_masks_all.append(ibatch_labels.cpu().detach())
                    del  ibatch_imgs, ibatch_labels,outputs_query,ataskmodel
                    torch.cuda.empty_cache()
                    ibatch_labels, ibatch_imgs,ataskmodel = None, None, None
                    gc.collect()
                ataskloss = torch.div(torch.sum(torch.stack(ataskloss)),tnumsample)
                outer_loss_alltasks.append(ataskloss)
                outputs_query = torch.cat(masks_pred_all)
                labels_query = torch.cat(true_masks_all)
                outputs_query_allF = torch.softmax(outputs_query, dim=1)
                outputs_query_cls = torch.argmax(outputs_query_allF, axis=1, keepdim=True)
                outputs_query_cls = torch.squeeze(outputs_query_cls)
                outputs_query_cls = torch.stack([1 - outputs_query_cls, outputs_query_cls], dim=1)
                outputs_query_cls = outputs_query_cls[:, :, 10:-10, 10:-10]
                labelsdim = torch.squeeze(labels_query[:, :, 10:-10, 10:-10])
                self.compute_iou(outputs_query_cls.cpu().detach().numpy(), labelsdim.cpu().detach().numpy().astype(int))
                ataskmiou=self.compute_iou.get_miou()
                ataskacc=self.acc_t(outputs_query_cls.cpu(), labelsdim.cpu())
            outer_loss_allbatch = torch.mean(torch.stack(outer_loss_alltasks))
            outer_loss_meta.append(outer_loss_allbatch.item())
            accuracy_query_meta.append(ataskacc)
            miou_query_meta.append(ataskmiou)
            self._optimizer.zero_grad()
            outer_loss_allbatch.backward()
            self._optimizer.step()
            del outer_loss_allbatch
            torch.cuda.empty_cache()
            ibatch_labels, ibatch_imgs = None, None
            gc.collect()
      #  print('support: mean MIoU:',mious_support_batch,';acc:',accuracies_support_batch)
      #  print('query:mean MIoU:', miou_query_meta, 'acc:', accuracy_query_meta)
        accuracies_support = np.mean(accuracies_support_batch,axis=0)
        accuracy_query = np.mean(accuracy_query_meta)
        mious_support=np.mean(mious_support_batch,axis=0)
        miou_query=np.mean(miou_query_meta)
        outer_loss=np.mean(outer_loss_meta)
      #  print('total metrics: query iou1:',iou1_query,';query mean iou:',miou_query,'accuracy_query:',accuracy_query)
        return outer_loss,accuracies_support, accuracy_query,mious_support,miou_query

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        print(f'Starting training at iteration {self._start_train_step}.')
        bestmiou=-999
        for i_step, task_batch in enumerate(dataloader_meta_train,start=self._start_train_step):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query,mious_support,miou_query = (
                self._outer_step(task_batch, train=True)
            )
            del task_batch
            torch.cuda.empty_cache()
            task_batch=None
            gc.collect()
            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss:.3f}, '
                    f'pre-adaptation support miou: '
                    f'{mious_support[0]:.3f}, '
                    f'post-adaptation support miou: '
                    f'{mious_support[-1]:.3f}, '
                    f'post-adaptation query miou: '
                    f'{miou_query:.3f}, '
                    f'pre-adaptation support acc: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support acc: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query acc: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss, i_step)
                writer.add_scalar(
                    'train_miou/pre_adapt_support',
                    mious_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_miou/post_adapt_support',
                    mious_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_miou/post_adapt_query',
                    miou_query,
                    i_step
                )
                writer.add_scalar(
                    'train_acc/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_acc/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_acc/post_adapt_query',
                    accuracy_query,
                    i_step
                )
            if i_step % VAL_INTERVAL == 0:
                losses = []
                miou_pre_adapt_support = []
                miou_post_adapt_support = []
                miou_post_adapt_query = []
                acc_pre_adapt_support = []
                acc_post_adapt_support = []
                acc_post_adapt_query = []
                for val_task_batch in dataloader_meta_val:
                    outer_loss, accuracies_support, accuracy_query,mious_support,miou_query  = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss)
                    miou_pre_adapt_support.append(mious_support[0])
                    miou_post_adapt_support.append(mious_support[-1])
                    miou_post_adapt_query.append(miou_query)
                    acc_pre_adapt_support.append(accuracies_support[0])
                    acc_post_adapt_support.append(accuracies_support[-1])
                    acc_post_adapt_query.append(accuracy_query)
                    del val_task_batch
                    torch.cuda.empty_cache()
                    val_task_batch = None
                    gc.collect()
                loss = np.mean(losses)
                miou_pre_adapt_support = np.mean(miou_pre_adapt_support)
                miou_post_adapt_support = np.mean(miou_post_adapt_support)
                miou_post_adapt_query = np.mean(miou_post_adapt_query)
                acc_pre_adapt_support = np.mean(acc_pre_adapt_support)
                acc_post_adapt_support = np.mean(acc_post_adapt_support)
                acc_post_adapt_query = np.mean(acc_post_adapt_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support miou: '
                    f'{miou_pre_adapt_support:.3f}, '
                    f'post-adaptation support miou: '
                    f'{miou_post_adapt_support:.3f}, '
                    f'post-adaptation query miou: '
                    f'{miou_post_adapt_query:.3f}, '
                    f'pre-adaptation support acc: '
                    f'{acc_pre_adapt_support:.3f}, '
                    f'post-adaptation support acc: '
                    f'{acc_post_adapt_support:.3f}, '
                    f'post-adaptation query acc: '
                    f'{acc_post_adapt_query:.3f} '
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar('val_miou/pre_adapt_support',miou_pre_adapt_support,i_step)
                writer.add_scalar('val_miou/post_adapt_support',miou_post_adapt_support,i_step)
                writer.add_scalar('val_miou/post_adapt_query',miou_post_adapt_query, i_step)
                writer.add_scalar('val_acc/pre_adapt_support',acc_pre_adapt_support,i_step)
                writer.add_scalar('val_acc/post_adapt_support',acc_post_adapt_support,i_step)
                writer.add_scalar('val_acc/post_adapt_query',acc_post_adapt_query, i_step)
            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)
            if bestmiou < miou_post_adapt_query:
                bestmiou = miou_post_adapt_query
                bestmetrics = pd.DataFrame({'itep': i_step, 'miou_pre_adapt_support': miou_pre_adapt_support,
                                            'miou_post_adapt_support': miou_post_adapt_support,
                                            'miou_post_adapt_query': miou_post_adapt_query,
                                            'acc_pre_adapt_support': acc_pre_adapt_support,
                                            'acc_post_adapt_support': acc_post_adapt_support,
                                            'acc_post_adapt_query': acc_post_adapt_query}, index=[i_step])
                tfl = self._log_dir + '/bestmetametric.csv'
                bestmetrics.to_csv(tfl, index=False)
                self._save(-1)

    def test(self, dataloader_test):
        ious1 = []
        for task_batch in dataloader_test:
            _, _, accuracy_query, _, _, iou1_query, miou_query =\
                      self._outer_step(task_batch, train=False)
            ious1.append(iou1_query)
        mean = np.mean(ious1)
        std = np.std(ious1)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'IOU1 over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )
    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

def main():
    args = docopt(__doc__)
    print(args)

    seed = int(args['--seed'])
    torch.manual_seed(seed)
    gpu = int(args['--gpu'])
    if gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:' + str(gpu))
    num_train_iterations = int(args['--num-train-iterations'])
    batch_size = int(args['--batch-size'])
    checkpoint_step = int(args['--checkpoint-step'])
    num_group = int(args['--num-group'])
    num_support = int(args['--num-support'])
    num_query = int(args['--num-query'])
    num_workers = int(args['--num-workers'])
    num_inner_steps = int(args['--num-inner-steps'])
    init_inner_lr = float(args['--init-inner-lr'])
    learn_inner_lrs = args['--learn-inner-lrs']
    iloop_batch = int(args['--iloop-batch'])
    oloop_batch = int(args['--oloop-batch'])
    num_outer_steps = int(args['--num-outer-steps'])
    outer_lr = float(args['--outer-lr'])

    log_dir = args['--log-dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    configDf = pd.DataFrame.from_dict([args])
    configDf.to_csv(log_dir + "/config.csv", header=True, index=False)

    modelconfig = {'n_channels': 3, 'n_classes': 2, 'encoders': [16, 32, 64, 128, 256, 512,1024],
            'bilinear': True}
    bunet=UNet(**modelconfig)
    maml = MAMLSeg(bunet,device,num_inner_steps=num_inner_steps,num_outer_steps=num_outer_steps,\
                   init_inner_lr=init_inner_lr,learn_inner_lrs=learn_inner_lrs,log_dir=log_dir,\
                   outer_lr=outer_lr,iloop_batch=iloop_batch,oloop_batch=oloop_batch)
    num_training_tasks = batch_size * (num_train_iterations -checkpoint_step - 1)

    dataloader_meta_train = get_gid5_dataloader(
        'train',
        batch_size,
        num_group,
        num_support,
        num_query,
        num_training_tasks,
        num_workers
    )
    dataloader_meta_val = get_gid5_dataloader(
        'val',
        batch_size,
        num_group,
        num_support,
        num_query,
        batch_size*4,
        num_workers
    )
    log_dir=log_dir
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    maml.train(
        dataloader_meta_train,
        dataloader_meta_val,
        writer
    )

if __name__=='__main__':
    main()

