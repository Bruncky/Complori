import pandas as pd

def get_data():
    data = pd.read_csv('data/train.csv')

    data.drop(columns = ['Id'], inplace = True)
    data.MSSubClass = data.MSSubClass.apply(str)

    return data
