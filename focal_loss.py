import torch
import torch.nn as nn
import torch.nn.functional as F 
import config as cfg 
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self,):
        super(FocalLoss, self).__init__()
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets, gamma=2):        
        tmp_input = inputs.clone()
        reverse_input = 1 - inputs
        pt = torch.where(targets==1, tmp_input, reverse_input)
        weights = torch.FloatTensor(cfg.attr_loss_weight) * torch.pow((1 - pt), gamma)
        weights = Variable(weights, requires_grad = False)
        loss = F.binary_cross_entropy(inputs.to(self.device), targets.type(torch.FloatTensor).to(self.device), weight=weights)
    
        if cfg.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def test():
    device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss()
    inputs = torch.rand([4,40])
    targets = torch.rand([4,40])
    loss = criterion(inputs, targets)
    # loss.backward()
    print(loss)

test()