import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_accuracy(acc_df):
    # get accuracy
    acc_list = acc_df.iloc[:, -1].tolist()

    # get attributes
    attrs_list = acc_df.iloc[:, 0].tolist()
    plt.figure(figsize=(12, 8))
    plt.plot(attrs_list, acc_list, marker='o', color='b')

    plt.xlabel(xlabel='attribute')
    plt.ylabel(ylabel='acc')
    plt.title(label='The accuracy of 40 attributes')

    plt.xticks(rotation = 90)
    plt.grid(True)
    # fig.savefig("test.png")
    plt.show()

def show_loss(loss_df):
    # get accuracy
    loss_list = loss_df.iloc[:, -1].tolist()

    # get attributes
    epoch_list = [i + 1 for i in range(len(loss_list))]

    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, loss_list, marker='o', color='b')

    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='loss')
    plt.title(label='The traininng loss')

    plt.xticks(rotation = 90)
    plt.grid(True)
    # fig.savefig("test.png")
    plt.show()


if __name__ == "__main__":
   
    acc_csv = './model/Resnet101-accuracy.csv'
    acc_df = pd.read_csv(acc_csv)
    show_accuracy(acc_df)
    print(acc_df.describe().T)

    loss_csv = './model/Resnet101-losses.csv'
    loss_df = pd.read_csv(loss_csv)
    show_loss(loss_df)
    