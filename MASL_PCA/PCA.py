'''
Created on 08 feb 2018

@author: marco
'''

if __name__ == '__main__':
    pass

import pandas as pd

# download dataset
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

# scelgo solamente le colonne con i valori di interesse
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class'] 
df.dropna(how="all", inplace=True) # Elimina i valori NA

df.tail()

print(df)
