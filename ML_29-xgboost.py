# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 04:41:55 2024

@author: ozan7
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('Churn_Modelling.csv')
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

onehotencoder = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = onehotencoder.fit_transform(X)
X = X[:,1:]




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

from xgboost import XGBClassifier
classifier =XGBClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)

















