# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:54:32 2024

@author: ozan7

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('clusteringveri.csv')

X = veriler.iloc[:,2:4].values

#kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_)

sonuclar = []
#cluster sayısını arttıyoruz 1 den 10 a kadar 
#random state verme sebebimiz center'ları dağıtmasın ve aynı yerde olsun diye random state değeri verdik.Aynı başlangıç değerlerine sahip olması için kısaca.
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) #kmeans ın ne kadar başarılı olduğunu gösterir.
    
plt.plot(range(1,11),sonuclar)
plt.show()
