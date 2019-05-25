import torch
import torch.nn as nn
import torch.nn.functional as F 
import config as cfg 
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self,):
        super(FocalLoss, self).__init__()
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets):        
        gpu_targets = targets.cuda()
        alpha_factor = torch.ones(gpu_targets.shape).cuda() * cfg.focal_loss_alpha
        alpha_factor = torch.where(torch.eq(gpu_targets, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(gpu_targets, 1), 1. - inputs, inputs)
        focal_weight = alpha_factor * torch.pow(focal_weight, cfg.focal_loss_gamma)
        targets = targets.type(torch.FloatTensor)
        inputs = inputs.cuda()
        targets = targets.cuda()
        bce = F.binary_cross_entropy(inputs, targets)
        focal_weight = focal_weight.cuda()
        cls_loss = focal_weight * bce
        return cls_loss.sum()
