import time
import math 
import numpy as np
from numpy import random
import os
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# import cv2 

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

def show_average_eval_acc(eval_df, save_path):
    mean_acc_list = eval_df.describe().loc['mean'].tolist()
    epoch_list = [i + 1 for i in range(len(mean_acc_list))]
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, mean_acc_list, marker='o', color='b')
    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='validation accuracy')
    plt.title(label='The validation loss')
    plt.xticks(rotation = 0)
    plt.grid(True)
    plt.savefig(save_path)

def show_loss(loss_df, save_path):
    # get loss value
    loss_list = loss_df.iloc[:, -1].tolist()
    # get epoches
    epoch_list = [i + 1 for i in range(len(loss_list))]

    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, loss_list, marker='o', color='b')
    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='loss')
    plt.title(label='The traininng loss')
    plt.xticks(rotation = 0)
    plt.grid(True)
    plt.savefig(save_path)

def show_mean_test_acc(test_df, save_path):
    
    test_acc_list = test_df.loc['accuracy']
    attr_list = test_df.columns.tolist()
    plt.figure(figsize=(12, 8))
    plt.plot(attrs_list, test_acc_list, marker='o', color='b')
    plt.xlabel(xlabel='attribute')
    plt.ylabel(ylabel='acc')
    plt.title(label='The test accuracy of 40 attributes')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.savefig(save_path)
    print("average test accuracy: {}".format(sum(test_acc_list)/len(test_acc_list)))