import time
import matplotlib.pyplot as plt
import math 
import numpy as np

"""
class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
"""

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