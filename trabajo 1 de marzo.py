from sklearn import datasets
import numpy as np
import lineal_regression as lr
import matplotlib.pyplot as plt
data_set=datasets.load_breast_cancer()
X=data_set['data']
y=data_set['target']
# Corrige el error de que los valores de X estén muy alejados
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
ll=lr.LinearRegression()
iteraciones,errores= ll.fit(X,y)
plt.plot(iteraciones,errores)
plt.show()

