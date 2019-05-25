import pandas as pd

"""
v3 - v6版本的度量指标 FN 错当成 TN，TN错当成FN，再交换csv属性名TN，FN之后，执行以下代码，重新计算召回率和准确率和F1
"""

if __name__ == "__main__":
    matrix = '../result/v6.1-se_resnet101-confusion_matrix.csv'
    matrix_df = pd.read_csv(matrix)
    print(matrix_df)
    matrix_df['precision'] = matrix_df['TP'] / (matrix_df['TP'] + matrix_df['FP'])
    matrix_df['recall'] = matrix_df['TP'] / (matrix_df['TP'] + matrix_df['FN'])
    matrix_df['F1'] = 2 * matrix_df['precision'] * matrix_df['recall'] / (matrix_df['precision'] + matrix_df['recall'])
    print(matrix_df.describe())
