import time
import matplotlib.pyplot as plt
import math 
import numpy as np
from numpy import random
import os
import torch
import cv2 
def timeSince(since):
    """
    compute the training/evaluate time
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%d h %d m %d s'%(h, m, s)

# show curve of loss or accuracy
def show_curve(self, ys, title):
    x = np.array([i for i in range(len(ys))])
    y = np.array(ys)
    plt.figure()
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{} value'.format(title))
    plt.show()
    plt.savefig("{}.jpg".format(title))

# ------------------------------------------------ # 
# make sure the same results with same params in different running time.
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
