import numpy as np
import torch
#from medpy import metric
import torch.nn as nn
#from PIL import Image
#from torchvision import transforms
#import SimpleITK as sitk
#from scipy.ndimage import zoom
from torch.nn import functional as F

class weiMSELoss(nn.Module):
    def __init__(self, bsize=10,size_average=None):
        super(weiMSELoss, self).__init__()
        self.size_average=size_average
        self.bsize=bsize

    def forward(self, inputs, targets,weights=None):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs=inputs[:,self.bsize:-self.bsize,self.bsize:-self.bsize]
        conloss = F.mse_loss(inputs, targets, reduction='mean',size_average=self.size_average)
        return conloss

class weiMSELossWei(nn.Module):
    def __init__(self, bsize=10,size_average=None):
        super(weiMSELossWei, self).__init__()
        self.size_average=size_average
        self.bsize=bsize

    def forward(self, inputs, targets, weights=None):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs[:, self.bsize:-self.bsize, self.bsize:-self.bsize]
        inputs=torch.squeeze(inputs)
        targets=torch.squeeze(targets)
        floss=0
        for i,w in enumerate(weights):
            conloss=F.mse_loss(inputs[i], targets[i], reduction='mean',size_average=self.size_average)
            floss+=conloss*w
        return floss
#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self,size_average=True):
        super(DiceBCELoss, self).__init__()
        self.size_average=size_average

    def forward(self, inputs, targets,weights=None, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        f_loss = F.binary_cross_entropy(inputs, targets,reduction='mean',size_average=self.size_average)
        Dice_BCE = f_loss + dice_loss
        return Dice_BCE


class DiceBCELossWei(nn.Module):
    def __init__(self, size_average=True):
        super(DiceBCELossWei, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, weights=None, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        f_loss=0
        inputs=torch.squeeze(inputs)
        targets=torch.squeeze(targets)
        for i,w in enumerate(weights):
            aloss=F.binary_cross_entropy(inputs[i], targets[i], reduction='mean', size_average=self.size_average)
            f_loss+=aloss*w
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        Dice_BCE = f_loss + dice_loss
        return Dice_BCE
  #PyTorch
class DiceBLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
