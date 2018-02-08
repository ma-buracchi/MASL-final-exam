'''
Created on 08 feb 2018

@author: marco
'''

import math
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# download dataset
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

# scelgo solamente le colonne con i valori di interesse
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class'] 
df.dropna(how="all", inplace=True) # Elimina i valori NA

df.tail()

# print(df)

# X = tabella con valori, y = etichette
X = df.ix[:,0:4].values
y = df.ix[:,4].values

# print(X)
# print(y)

# creazione istogrammi, (decommentare from matplotlib import pyplot as plt)
# label_dict = {1: 'Iris-Setosa',
#               2: 'Iris-Versicolor',
#               3: 'Iris-Virgnica'}
# 
# feature_dict = {0: 'sepal length [cm]',
#                 1: 'sepal width [cm]',
#                 2: 'petal length [cm]',
#                 3: 'petal width [cm]'}
# 
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(8, 6))
#     for cnt in range(4):
#         plt.subplot(2, 2, cnt+1)
#         for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
#             plt.hist(X[y==lab, cnt],
#                      label=lab,
#                      bins=10,
#                      alpha=0.3,)
#         plt.xlabel(feature_dict[cnt])
#     plt.legend(loc='upper right', fancybox=True, fontsize=8)
# 
#     plt.tight_layout()
#     plt.show()
    
# normalizzazione dati 
X_std = StandardScaler().fit_transform(X)

#vettore delle medie e matrice di covarianza
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Matrice di covarianza calcolata: \n%s' %cov_mat)

# funzione di libreria
print('Matrice di covarianza NumPy: \n%s' %np.cov(X_std.T))

# calcolo autovalori e autovettori
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Autovettori: \n%s' %eig_vecs)
print('\nAutovalori: \n%s' %eig_vals)

if __name__ == '__main__':
    pass