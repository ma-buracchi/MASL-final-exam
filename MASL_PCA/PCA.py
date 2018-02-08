'''
Created on 08 feb 2018

@author: marco
'''

import math
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def separatore():
    print('############################################################')

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

# vettore delle medie e matrice di covarianza
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Matrice di covarianza calcolata: \n%s\n' %cov_mat)

# funzione di libreria
print('Matrice di covarianza NumPy: \n%s\n' %np.cov(X_std.T))
separatore()

# calcolo autovalori e autovettori su matrice di covarianza
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Autovettori cov: \n%s\n' %eig_vecs)
print('Autovalori cov: \n%s\n' %eig_vals)
separatore()

# calcolo autovalori ed autovettori su matrice di correlazione dati standardizzati
cor_mat1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
print('Autovettori corrSTD: \n%s\n' %eig_vecs)
print('Autovalori corrSTD: \n%s\n' %eig_vals)
separatore()

# calcolo autovalori ed autovettori su matrice di correlazione dati grezzi
cor_mat2 = np.corrcoef(X.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat2)
print('Autovettori corr: \n%s\n' %eig_vecs)
print('Autovalori corr: \n%s\n' %eig_vals)
separatore()

# decomposizione ai valori singolari
u,s,v = np.linalg.svd(X_std.T)
print('Autovettori SVD: \n%s\n' %u)
separatore()

if __name__ == '__main__':
    pass