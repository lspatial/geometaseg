import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
import pandas as pd
from datagid5.np_argumenter import Augmenter

NUM_TRAIN_GROUPS = 2
NUM_VAL_GROUPS = 2
NUM_TEST_GROUPS = 2
NUM_TOTAL_GOUPS = 6


class Gid5Dataset(dataset.Dataset):
    """Gid5Dataset dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """
    _BASE_PATH = '/geosampling/rs_gidcls5/cdist10_r256sc1_mcomplex_sum' # 'D:/wkspacedata/gmetacomplex/gid5/'

    def __init__(self, tcls,num_support, num_query,grpvar='m_all',mint=0.05,arguments=None,argrate=0):
        """Inits Gid5Dataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()
        self._num_support = num_support
        self._num_query = num_query
        self.tcls = tcls
        tpath =  self._BASE_PATH + '/patchall_cls' + str(tcls) + '_sum.csv' #self._BASE_PATH + '/patchall_cls' + str(tcls) + '_sum_adj.csv'
        assert os.path.exists(tpath),'Please ensure that the summary file exists!'
        self.moranSumTb=pd.read_csv(tpath)
#       self.moranSumTb=self.moranSumTb[self.moranSumTb['p_all']>=0.2]
        quantile = [1.0 / float(NUM_TOTAL_GOUPS) * i for i in range(NUM_TOTAL_GOUPS + 1)]
        labels = [i for i, k in enumerate(quantile) if i > 0]
        self.moranSumTb['mgroup'] = pd.qcut(self.moranSumTb[grpvar], q=quantile, labels=labels)
        self.grpvar=grpvar
        self.mint=mint
        self.augmenter=None
        if arguments is None:
            self.augmenter=None
        elif isinstance(arguments,bool) and  arguments:
            self.augmenter=Augmenter()
        elif isinstance(arguments,list):
            self.augmenter = Augmenter(arguments)
        self.argrate=argrate

    def __getitem__(self, group_index):
        """Constructs a task.
        Data for each class is sampled uniformly at random without replacement.
        Args:
            group_index (tuple[float]): group values that comprise the task
        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,height, width)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query, height, width)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []
        reverse_lab=random.random()
        for _,gindice in enumerate(group_index):
            selDf=self.moranSumTb[self.moranSumTb['mgroup']==gindice]
            np.set_printoptions(precision=15)
            weight=selDf['m_all'].values/np.sum(selDf['m_all'].values)
            sampled_file_paths=np.random.choice(selDf['tfl'].values,size=self._num_support \
                           + self._num_query,p=weight,replace=False)
            images,labels=[],[]
            for f_path in sampled_file_paths:
                #f_path = self._BASE_PATH + '/' + str(self.tcls)+'/'+ os.path.split(apath)[-1]
                assert os.path.exists(f_path),f_path + ' is not exists!'
                with  np.load(f_path, allow_pickle=True) as adata: 
                    mask = adata['orgmask']
                    img = adata['orgimg'].astype(np.float32)
#                mask = mask[:, :, None]
                    if reverse_lab>0.5:
                        mask = np.stack([mask,1-mask],-1)
                    else:
                        mask = np.stack([1-mask,mask],-1)
                if self.augmenter is not None:
                    img, mask = self.augmenter.apply_augmentations(img, mask, self.argrate)
                img = img / 256.0
                img = np.moveaxis(img, -1, 0)
                mask = np.moveaxis(mask, -1, 0)
                images.append(torch.tensor(img))
                labels.append(torch.tensor(mask))
            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend(labels[:self._num_support])
            labels_query.extend(labels[self._num_support:])
        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.stack(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.stack(labels_query)
        return images_support, labels_support, images_query, labels_query

class GID5Sampler(sampler.Sampler):
    def __init__(self, mgroups_index, num_way, num_tasks):
        """Inits GID5Sampler.
        Args:
            mgroups_index (list): summarized Moran's I values that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self.mgroups_index = mgroups_index
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self.mgroups_index ,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def identity(x):
    return x


def get_gid5_dataloader(
        split,
        batch_size,
        num_group,
        num_support,
        num_query,
        num_tasks_per_epoch,
        num_workers=2,
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_group (int): number of groups per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """
##Here train, validation and test ensure no overlap.
    np.random.seed(1)
    gindex=[i for i in range(1,NUM_TOTAL_GOUPS+1)]
    np.random.shuffle(gindex)
    if split == 'train':
        mgindex = [gindex[i] for i in range(NUM_TRAIN_GROUPS)]
    elif split == 'val':
        mgindex = [gindex[i] for i in range(NUM_TRAIN_GROUPS,NUM_TRAIN_GROUPS+NUM_VAL_GROUPS)]
    elif split == 'test':
        mgindex = range(
            NUM_TRAIN_GROUPS+NUM_VAL_GROUPS,
            NUM_TRAIN_GROUPS+NUM_VAL_GROUPS + NUM_TEST_GROUPS
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=Gid5Dataset(2,num_support, num_query,arguments=True,argrate=0.5),
        batch_size=batch_size,
        sampler=GID5Sampler(mgindex, num_group, num_tasks_per_epoch),
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
