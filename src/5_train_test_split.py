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

df_path = root.DIR_DATA_STAGE + 'df_final_infouser.csv'
df_infousers = pd.read_csv(df_path)


# train test split
X = df_infousers.drop('Cuotas en mora', axis=1)
y = df_infousers['Cuotas en mora']

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

path_infocreditos = root.DIR_DATA_STAGE + 'df_final_creditinfo.csv'
df_infocreditos = pd.read_csv(path_infocreditos)

X = df_infocreditos.drop('mora', axis=1)
y = df_infocreditos['mora']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

file_path = root.DIR_DATA_STAGE + 'train_creditinfo.csv'
df_train.to_csv(file_path, index=False)
file_path = root.DIR_DATA_STAGE + 'test_creditinfo.csv'
df_test.to_csv(file_path, index=False)