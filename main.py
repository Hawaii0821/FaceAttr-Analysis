from solver import Solver
import os
import random
import numpy as np
import torch
import pandas as pd
import argparse
from utils import seed_everything

parser = argparse.ArgumentParser(description='FaceAtrr')
parser.add_argument('--model_type', choices=['Resnet101','Resnet152','gc_resnet101','se_resnet101', 
'densenet121', 'sge_resnet101', 'sk_resnet101'], default='Resnet101')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--epoches', default=100, type=int, help='epoches')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optim_type', choices=['SGD','Adam'], default='SGD')
parser.add_argument('--pretrained', action='store_true', default=True)
parser.add_argument("--loss_type", choices=['BCE_loss', 'focal_loss'], default='BCE_loss')
parser.add_argument("--exp_version",type=str, default="v7")
parser.add_argument("--load_model_path", default="", type=str)
args = parser.parse_args()

epoches = args.epoches
batch_size = args.batch_size
learning_rate = args.learning_rate
model_type = args.model_type
optim_type = args.optim_type
momentum = args.momentum
pretrained = args.pretrained
loss_type = args.loss_type
exp_version = args.exp_version
model_path = args.load_model_path

#--------------- exe ----------------------------- #
if __name__ == "__main__":
    seed_everything()

    # too more params to send.... not a good way....use the config.py to improve it
    solver = Solver(epoches=epoches, batch_size=batch_size, learning_rate=learning_rate, model_type=model_type,
                    optim_type=optim_type, momentum=momentum, pretrained=pretrained, loss_type=loss_type,
                    exp_version=exp_version)
    try:
        solver.fit(model_path=model_path)
        solver.test_speed(256)

    except KeyboardInterrupt:
        print("early stop...")
        print("save the model dict....")
        solver.save_model_dict(exp_version+"_"+model_path + "_earlystop.pth")
