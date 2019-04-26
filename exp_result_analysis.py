import pandas 
import matplotlib


if __name__ == "__main__":
    result_csv = './model/Resnet101-accuracy.csv'
    df = pandas.read_csv(result_csv, index_col=False)
    print(df.head(10))
    print(df.describe())
    des = df.describe()
    print(des)