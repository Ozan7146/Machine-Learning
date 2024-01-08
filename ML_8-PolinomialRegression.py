import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('maaslar.csv')
#dataframe den aldığımız için values ekini eklememiz lazım
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#linear regression için
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X))
#Burada x ve x in linear regresion daki karşılığını yazdırdık. 
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#polynomial regresyona çevirdik 
poly_reg = PolynomialFeatures(degree = 2)

#polinomal regresyonu linear reg şeklinde yazdık
x_poly = poly_reg.fit_transform(X)
print(x_poly)

#ogrenme yeri
lin_reg2 = LinearRegression()
#x polyi kullanarak y yi öğren
lin_reg2.fit(x_poly,y)
plt.scatter(X, Y)
#burda poly.reg yapamamızın sebebi yukarda bu değeri 2.dereceden tanımladık
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()







