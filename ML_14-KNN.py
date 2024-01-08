# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:06:30 2023

@author: ozan7
Logistic Regression
"""



#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,2:3].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train) #eğitim kısmı

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm) 

#KNN Algorithm

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,metric= 'minkowski')
knn.fit(X_train,y_train) #eğitme kısmı
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)





 










