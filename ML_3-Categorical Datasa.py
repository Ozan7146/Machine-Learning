# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:36:53 2023

@author: ozan7
bilkav->eğitimler->udemy->A dan Z ye makine öğrenmesi içinde gerekli veriler var.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv("MissingDatas.csv")

from sklearn.impute import SimpleImputer

#kategorik verileri sayısala çevirip ml nin anlayacağı hale getirmek
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#dataframe oluşturduk istediğimiz değerlere göre
sonuc = pd.DataFrame(data=ulke, index = range(22), columns =['fr','tr','us'])
print(sonuc)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc2 = pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc2)

#axis=1 dersek alt alta ekleme yerine yan yana ekleme yapar
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
