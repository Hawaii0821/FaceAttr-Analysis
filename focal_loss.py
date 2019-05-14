import torch
import torch.nn as nn
import torch.nn.functional as F 
import config as cfg 
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self,):
        super(FocalLoss, self).__init__()
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets, alpha_factor, focal_weight, gamma=2):        
        alpha_factor = torch.where(torch.eq(targets, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1), 1. - inputs, inputs)
        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
        targets = targets.type(torch.FloatTensor)
        bce = -(targets * torch.log(inputs) + (1. - targets) * torch.log(1. - inputs))
        cls_loss = focal_weight * bce
        return cls_loss.mean()

