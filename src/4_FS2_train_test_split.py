# import librerias de python
import os
import sys
# importar librerias para el procesamiento
import pandas as pd
from sklearn.model_selection import train_test_split

# carga de datos
current_dir = os.getcwd() # Obtener la ruta del directorio actual del notebook
ROOT_PATH = os.path.dirname(current_dir) # Obtener la ruta del directorio superior
sys.path.insert(1, ROOT_PATH) # Insertar la ruta en sys.path

import root # Importar el módulo root

df_path = root.DIR_DATA_STAGE + 'df_imputado.csv'
df = pd.read_csv(df_path)


# eliminar columnas por alta correlacion entre ellas
num_cols_to_drop = ['INT CORRIENTE']
df_final = df.drop(num_cols_to_drop, axis=1)

# train test split
X = df_final.drop('Cuotas en mora', axis=1)
y = df_final['Cuotas en mora']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# guardar los datos

# infousers
file_path = root.DIR_DATA_STAGE + 'train_infousers.csv'
df_train.to_csv(file_path, index=False)

file_path = root.DIR_DATA_STAGE + 'test_infousers.csv'
df_test.to_csv(file_path, index=False)

#credit info 

path_infocreditos = root.DIR_DATA_RAW + 'db_raw_creditinfo.csv'
df_infocreditos = pd.read_csv(path_infocreditos)

df_infocreditos['PRÓXIMA FECHA PAGO'] = pd.to_datetime(df_infocreditos['PRÓXIMA FECHA PAGO'])
df_infocreditos['mes'] = df_infocreditos['PRÓXIMA FECHA PAGO'].dt.month

df_infocreditos.drop(columns=['PRÓXIMA FECHA PAGO','INT CORRIENTE'], inplace=True)

X = df_infocreditos.drop('mora', axis=1)
y = df_infocreditos['mora']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

file_path = root.DIR_DATA_STAGE + 'train_creditinfo.csv'
df_train.to_csv(file_path, index=False)
file_path = root.DIR_DATA_STAGE + 'test_creditinfo.csv'
df_test.to_csv(file_path, index=False)