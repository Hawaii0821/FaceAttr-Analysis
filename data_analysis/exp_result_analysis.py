import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show_eval_accuracy(eval_df, save_path):
    mean_acc_list = eval_df.describe().loc['mean'].tolist()
    epoch_list = [i + 1 for i in range(len(mean_acc_list))]
    plt.figure(figsize=(12, 10))
    plt.plot(epoch_list, mean_acc_list, marker='o', color='b')
    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='validation accuracy')
    plt.title(label='The validation accuracy')
    plt.xticks(rotation = 0)
    plt.grid(True)
    plt.savefig(save_path)

def show_test_accuracy(acc_df, save_file_name):
    acc_list = acc_df.iloc[0, 1:].tolist()
    attrs_list = acc_df.columns[1:].tolist()

    plt.figure(figsize=(18, 12))
    plt.plot(attrs_list, acc_list, marker='o', color='b')
    plt.xlabel(xlabel='attribute')
    plt.ylabel(ylabel='acc')
    plt.title(label='The test accuracy of 40 attributes')
    plt.xticks(rotation=60)
    plt.grid(True)
    plt.savefig(save_file_name)
    return sum(acc_list)/len(acc_list)

def show_loss(loss_df, save_file_name):
    loss_list = loss_df.iloc[:, -1].tolist()
    epoch_list = [i + 1 for i in range(len(loss_list))]
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, loss_list, marker='o', color='b')
    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='loss')
    plt.title(label='The traininng loss')
    plt.xticks(rotation = 60)
    plt.grid(True)
    plt.savefig(save_file_name)
    # plt.show()


if __name__ == "__main__":

    eval_acc_csv = '../result/v15-shuffle_netv2-eval_accuracy.csv'
    eval_acc_df = pd.read_csv(eval_acc_csv)
    show_eval_accuracy(eval_acc_df, eval_acc_csv.replace('.csv', '.png'))


    # print("ok!")
    test_acc_csv = '../result/v15-shuffle_netv2-test_accuracy.csv' 
    acc_df = pd.read_csv(test_acc_csv)
    acc = show_test_accuracy(acc_df, test_acc_csv.replace('.csv', '.png'))
    aver_acc = show_test_accuracy(acc_df, test_acc_csv.replace('.csv', '.png'))
    
    loss_csv = '../result/v15-shuffle_netv2-losses.csv'
    loss_df = pd.read_csv(loss_csv)
    show_loss(loss_df, loss_csv.replace('.csv', '.png'))
    
    matrix = '../result/v15-shuffle_netv2-confusion_matrix.csv'
    matrix_df = pd.read_csv(matrix)

    
    matrix_df['precision'] = matrix_df['TP'] / (matrix_df['TP'] + matrix_df['FP'])
    matrix_df['recall'] = matrix_df['TP'] / (matrix_df['TP'] + matrix_df['FN'])
    matrix_df['F1'] = 2 * matrix_df['precision'] * matrix_df['recall'] / (matrix_df['precision'] + matrix_df['recall'])
    print(matrix_df.describe())
    print("average accuracy: {}".format(acc))


    """
    all_attrs = ['Attractive','Bald', 'Bangs', 'Black_Hair','Blond_Hair','Brown_Hair', 'Chubby', 'Double_Chin', 
                'Eyeglasses', 'Goatee', 'Gray_Hair','Male', 'Mustache', 'Pale_Skin', 'Receding_Hairline',  
                'Smiling', 'Wearing_Hat'
    ]

    v12 gc:95.05
    v10 sk:95.28
    v9  sge: 94.97
    v8 94.97
    v7.1 95.28
    v6.2 95.26
    v5.2 95.26
    v2.1 94.96
    """