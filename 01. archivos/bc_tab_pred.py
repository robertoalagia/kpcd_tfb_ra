
# Importar las librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
from sklearn import preprocessing
warnings.filterwarnings('ignore') #ignorar warnings

# Carga de datos
df = pd.read_csv('breast-cancer_train.csv', sep=';', decimal='.')

# Categorizamos la columna 'diagnosis'
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Eliminamos las columnas que están correlacionadas por encima del 90%
matrix_abs = df.corr().abs()
mask = np.triu(np.ones_like(matrix_abs, dtype=bool))

triangular_df = matrix_abs.mask(mask) #matriz triangular superior con ceros en la diagonal y por debajo
to_drop = [x for x in triangular_df.columns if any(triangular_df[x]>0.9)] #se eliminan aquellas características que tengan más de 0.9 de correlación.

df.drop('id', axis=1, inplace=True)
df = df.drop(to_drop, axis=1)

# Rellenamos valores faltantes (por si aplicaría en el futuro)
df = df.fillna(df.mean())

# Ponemos la columna 'diagnosis' en la 1º posición
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('diagnosis')))
df = df[cols]

# Dataset de train
data_train = df.values
y_train = data_train[:,0:1]     # nos quedamos con la 1ª columna, price
X_train = data_train[:,1:]      # nos quedamos con el resto

# Escalamos (con los datos de train)
scaler = preprocessing.StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)
