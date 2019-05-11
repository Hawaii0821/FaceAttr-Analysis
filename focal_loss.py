import torch
import torch.nn as nn
import torch.nn.functional as F 
import config as cfg 

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self,):
        super(FocalLoss, self).__init__()
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets, gamma=2, alpha=None):
        total_loss = torch.tensor([0.0]).to(self.device)
        batch_size = inputs.size(0)
        
        self.attr_loss_weight = torch.tensor(cfg.attr_loss_weight).to(self.device)
        for i in range(batch_size):
            attr_list = []

            # terrible tensor
            for j, attr in enumerate(self.selected_attrs):
                attr_list.append(targets[j][i])
            attr_tensor = torch.tensor(attr_list)

            weights = self.attr_loss_weight.type(torch.FloatTensor) * torch.pow((1 - inputs[i]), gamma)
            weights.to(self.device)
            loss = F.binary_cross_entropy(inputs[i].to(self.device),  
                                          attr_tensor.type(torch.FloatTensor).to(self.device), 
                                          weight=weights)
            total_loss += loss

        if cfg.size_average:
            loss = total_loss.mean()
        else:
            loss = total_loss.sum()
        return loss