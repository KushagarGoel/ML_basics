# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:45:46 2020

@author: kushagar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values

plt.scatter(X,y)

from sklearn.model_selection import train_test_split

X_train,X_test , y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test,y_test, color = "red")
plt.plot(X_test, y_pred)
plt.show()