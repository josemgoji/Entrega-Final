# importar librerias de python
import os
import sys
# importar librerias para el procesamiento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
# librerias Imputacion y evaluacion
from sklearn.linear_model import LinearRegression 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
#libreria para codificacion
# cargar los datos
current_dir = os.getcwd() # Obtener la ruta del directorio actual del notebook
ROOT_PATH = os.path.dirname(current_dir) # Obtener la ruta del directorio superior
sys.path.insert(1, ROOT_PATH) # Insertar la ruta en sys.path

import root # Importar el módulo root

file_path = root.DIR_DATA_STAGE + 'db_pross_info_users.csv'
df = pd.read_csv(file_path)

# eliminar columnas con mas de 30% de valores nulos
df.drop(['PERSONAS A CARGO','NUMERO DE HIJOS','INGRESOS ADICIONALES'], axis=1, inplace=True)

# imputacion categorica
cols_with_nan = [col for col in df.columns if df[col].isnull().any()]
num_cols_nan = [col for col in df[cols_with_nan].select_dtypes(exclude=['object'])]
cat_cols_nan = [col for col in df[cols_with_nan].select_dtypes(include=['object'])]
cols = ['ESTRATO']
num_cols_nan = [col for col in num_cols_nan if col not in cols]
cat_cols_nan.extend(cols)


imputer = SimpleImputer(strategy='most_frequent')

#imputa las columnas
df[cat_cols_nan] = imputer.fit_transform(df[cat_cols_nan])

# imputacion numerica
df['INGRESOS MENSUALES'] = df['INGRESOS MENSUALES']/1000000
df['GASTOS MENSUALES'] = df['GASTOS MENSUALES']/1000000
df['CAPITAL'] = df['CAPITAL']/1000000

# decteccion de ouliers numericos

num_cols = [col for col in df.select_dtypes(exclude=['object'])]
cols = ['Cuotas en mora']
num_cols = [col for col in num_cols if col not in cols]
num_cols

#distancia de mahalanobish
df2 = df[num_cols].copy()
df2 = df2.apply(lambda col: col.fillna(col.median()), axis=0)
mu = df2.mean().values ### Calculando la media de los datos
Sigma_inv = np.linalg.inv(df2.cov().values) #### Calculando la inversa de la mtriz de covarianzas

centered = df2 - mu  ### Centrando los datos
centered = centered.to_numpy() ## par hacer multiplicacion de matrices es mejor llevar los dataframes a arrays

MD2 = np.matmul(np.matmul(centered, Sigma_inv),centered.transpose()) ### Usando la formula de la distancia de mahalanobis

MD2 = np.diag(MD2) ## Extrayendo la diagonal que mide la distancia al centroide de los puntos

from scipy.stats import chi2
alpha = 0.01
cut_off = chi2.ppf(1-alpha, len(df.columns)) ### Chi cuadrado con 1-alpha % y p grados de libertad
##### los grados de libertad p son la cantidad de variables es decir el largo de el vector de nombres de las columans

df2['MD2'] = MD2
df2["out"] = "0"
df2.loc[MD2 > cut_off, "out"] = "1"

outliers_idx = df2[df2["out"] == "1"].index
df_clean = df.drop(outliers_idx)

#truncamiento de outliers
def clean_univariate_sample(df, column_name):
    # Extraer la columna del DataFrame
    x = df[column_name].dropna()  # Remover NaN para evitar problemas en el cálculo de percentiles

    # Calcular los percentiles y el rango intercuartílico
    Q1, Q3 = np.percentile(x, [25, 75])
    IQR = Q3 - Q1

    # Calcular los límites para identificar valores atípicos
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    # Filtrar y actualizar el DataFrame original sin outliers
    df[column_name] = df[column_name].apply(lambda y: lim_inf if y <= lim_inf else y)
    df[column_name] = df[column_name].apply(lambda y: lim_sup if y >= lim_sup else y)

    return df

df_clean2 = df_clean.copy()

df_clean2 = clean_univariate_sample(df_clean2, 'INGRESOS MENSUALES')
df_clean2 = clean_univariate_sample(df_clean2, 'meses_transcurridos')
df_clean2 = clean_univariate_sample(df_clean2, 'GASTOS MENSUALES')

#impputacion
imputer = SimpleImputer(strategy='median')
cols_to_impute = ['GASTOS MENSUALES','meses_transcurridos']

#imputa las columnas
df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])


X = df_clean.reset_index(drop=True)
X_num = X[num_cols]
X_cat = X.drop(num_cols, axis=1)

X = df_clean.reset_index(drop=True)
X_num = X[num_cols]
X_cat = X.drop(num_cols, axis=1)

# Crear el LinearRegression
linear_regressor = LinearRegression()

# Configurar el IterativeImputer con regresión lineal
iterative_imputer_lr = IterativeImputer(estimator=linear_regressor, max_iter=10, random_state=0)
X_imputed = iterative_imputer_lr.fit_transform(X_num)
X_imputed = pd.DataFrame(X_imputed, columns=num_cols)

df_imputed = pd.concat([X_imputed, X_cat], axis=1)

## codificacion

### Codificacion fecha desembolso a meses
df_imputed['PRÓXIMA FECHA PAGO'] = pd.to_datetime(df_imputed['PRÓXIMA FECHA PAGO'], errors='coerce', format='%Y-%m-%d')
df_imputed['mes_de_pago'] = df_imputed['PRÓXIMA FECHA PAGO'].dt.month

df_imputed.drop(columns=['PRÓXIMA FECHA PAGO'], inplace=True)

cat_cols = [col for col in df.select_dtypes(include=['object'])]
cat_cols.remove('PRÓXIMA FECHA PAGO')
cat_cols.remove('ESTRATO')

df_encoded = pd.get_dummies(df_imputed, columns=cat_cols, drop_first=True,dtype=float)

# guardar los datos 
file_path = root.DIR_DATA_STAGE + 'df_imputado.csv'
df_encoded.to_csv(file_path, index=False)


