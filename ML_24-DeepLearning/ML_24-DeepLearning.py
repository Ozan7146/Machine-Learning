
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values



#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#3 Yapay Sinir ağı
import keras
from keras.models import Sequential #Yapay sinir ağı oluşturmak için kullandık
from keras.layers import Dense #Yapasy sinir ağımda katman oluşturmak için kullandık(nöron oluşturmak)

classifier = Sequential() #Boş bir yapay sinir ağımız var

#objeler ekliyoruz
#6 değeri aradaki gizli katman sayısını gösterir
classifier.add(Dense(6, init = 'uniform', activation = 'relu' , input_dim = 11))#uniform initialize etmeye yarar.input_dim girişte kaç bağımsız değişken olduğunu söyler
# 6 vermemizin sebebi girişte 11 değer var çıkışta 1 değer var topla 2 ye böl bu ortadaki katman sayısını verir.Bu bir yöntemdir.
#aktivasyon fonksiyonunu relu olarak seçtik yani rectifier oldu 

#ikinci gizli katmanı ekliyoruz
classifier.add(Dense(6, init = 'uniform', activation = 'relu')) 
#çıkış kısmını verdiks
classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

#Şuan giriş katmanında 11 nöron var.2 tane gizli katman var ve her birinde 6şar nöron var.çıkış katmanında ise 1 nöron var. 

#Sinir ağının(Neural Network'ün) nasıl çalışacağını keras kütüphanesi üzerinden pythona anlatıyoruz.
#Farklı optimizer'ları keras'ın dokümantosyonu üzerinden bakabilirsin
#Keras tensorflow üzerinden çalıştığı için bazen tensorflow dokümantasyonunu da okuyacaksın
#'adam' aslında stochastic gradient descendent'in bir versiyonu ve sinapsisler üzerindeki değerleri nasıl optimize edeceğimizi gösteriyor.
# The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )


#Makine öğrenmesi kısmını verdik
#epochs kaç aşamada öğreneceğidir.
classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test)

#0.5 in altıdaysa 0 üstündeyse 1 döndürür
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)














