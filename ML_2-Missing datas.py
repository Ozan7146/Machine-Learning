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
#print(veriler)
#boy = veriler[['boy']]
#print(boy)
#boykilo = veriler[['boy','kilo']]
#print(boykilo)

#x=10
#eksik veriler NaN olarak gözükür

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#yukarda hangi boşlukların doldurulacağını söyledik(NaN)
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
#yas kısmındaki sayıların ortalamasıyla NaN yerleri doldurduk

