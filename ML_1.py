# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:36:53 2023

@author: ozan7
bilkav->eğitimler->udemy->A dan Z ye makine öğrenmesi içinde gerekli veriler var.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv("veriler.csv")
print(veriler)
boy = veriler[['boy']]
print(boy)
boykilo = veriler[['boy','kilo']]
print(boykilo)

x=10






