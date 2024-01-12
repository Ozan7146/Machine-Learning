#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:03:45 2018

@author: sadievrenseker
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000
d = 10 
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d) #Rasgele bir sayı üretiliyor
    secilenler.append(ad) #hangi değerin ilanın seçildiğini gösteririr
    odul = veriler.values[n,ad] # Tıklanan ilan bizim daha önceden seçtiğimiz ilansa.verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
print(toplam)
plt.hist(secilenler)
plt.show()










