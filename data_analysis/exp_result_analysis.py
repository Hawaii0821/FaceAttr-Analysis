import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_accuracy(acc_df, save_file_name):
    # get attributes
    attrs_list = acc_df.columns[1:]
    # print(attrs_list)
    # get accuracy
    acc_list = acc_df.iloc[-1, 1:]
    print(acc_list)
    plt.figure(figsize=(12, 8))
    plt.plot(attrs_list, acc_list, marker='o', color='b')
    plt.xlabel(xlabel='attribute')
    plt.ylabel(ylabel='acc')
    plt.title(label='The accuracy of 40 attributes')
    plt.xticks(rotation = 90)
    plt.grid(True)
    plt.savefig(save_file_name)
    # plt.show()
    return sum(acc_list)/len(acc_list)

def show_loss(loss_df, save_file_name):
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
    plt.savefig(save_file_name)
    # plt.show()


if __name__ == "__main__":
    acc_csv = '../result/v3-gc_resnet101-accuracy.csv'
    acc_df = pd.read_csv(acc_csv)
    aver_acc = show_accuracy(acc_df, acc_csv.replace('.csv', '.png'))
    print(acc_df.describe())
   
    loss_csv = '../result/v3-gc_resnet101-losses.csv'
    loss_df = pd.read_csv(loss_csv)
    show_loss(loss_df, loss_csv.replace('.csv', '.png'))

    matrix = '../result/v3-gc_resnet101-confusion_matrix.csv'
    matrix_df = pd.read_csv(matrix)
    print(matrix_df)
    print("Average accuracy: {} ".format(aver_acc))
    precision = matrix_df['TP'].mean() / (matrix_df['TP'].mean() + matrix_df['FP'].mean())
    recall = matrix_df['TP'].mean() / (matrix_df['TP'].mean() + matrix_df['FN'].mean())
    print("macro precion:{}".format(precision))
    print('macro recall:{}'.format(recall))
    print("macro F1:{}".format(2*precision*recall / (precision + recall)))