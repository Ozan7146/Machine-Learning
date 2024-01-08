# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
#print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

'''
#verilerin olceklenmesi->verileri standartlaştırıyoruz
from sklearn.preprocessing import StandardScaler


sc=StandardScaler()
#train kısımlarını alıp modelimizi oluştrmak için kullanacağız
X_train = sc.fi t_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


#model inşası(linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#X_train den Y_train i öğrendi
lr.fit(X_train,Y_train)


#lr bir linear regresyon modelidir bu model X_test teki verileri kullanarak Y_test teki verileri tahmin etti 
prediction = lr.predict(X_test)
'''

#model inşası(linear regression)->verileri standardize etmeden yaptık 
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#X_train den Y_train i öğrendi
lr.fit(x_train,y_train)


#lr bir linear regresyon modelidir bu model X_test teki verileri kullanarak Y_test teki verileri tahmin etti 
#prediction ile Y_test i karşılaştırırsan tahminin doğruluk oranını görebilirsin 
prediction = lr.predict(x_test)

#Veri Görselleştirme
#indexe göre sırala
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
#plt.plot(x_test,prediction) #asagidakiyle ayni
plt.plot(x_test,lr.predict(x_test))
plt.title("aylara gore satis")
plt.xlabel("aylar")
plt.ylabel("satislar")





