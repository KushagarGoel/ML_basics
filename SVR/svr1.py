# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:30:46 2020

@author: kushagar
"""


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)
arr = np.array(6.5)
arr = arr.reshape(1,-1)
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(arr)))

plt.scatter(X, y, color="red")
plt.plot(X,regressor.predict(X))
plt.show()