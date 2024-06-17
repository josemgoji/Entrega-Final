# import librerias de python
import os
import sys
# importar librerias para el procesamiento
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# carga de datos
current_dir = os.getcwd() # Obtener la ruta del directorio actual del notebook
ROOT_PATH = os.path.dirname(current_dir) # Obtener la ruta del directorio superior
sys.path.insert(1, ROOT_PATH) # Insertar la ruta en sys.path

import root # Importar el módulo root

# infouser

df_path = root.DIR_DATA_STAGE + 'df_imputado.csv'
df = pd.read_csv(df_path)


# eliminar columnas por alta correlacion entre ellas
num_cols_to_drop = ['INT CORRIENTE']

# eliminar por correlacion de variables categoricas
binary_columns = df.columns[(df.nunique() == 2)]
binary_columns = binary_columns.tolist() + ['ESTRATO', 'mes_de_pago']

df_cat = df[binary_columns]
X = df_cat.drop('Cuotas en mora', axis=1)
y = df_cat['Cuotas en mora']

chi_scores = chi2(X, y)
p_values = pd.Series(chi_scores[1], index=X.columns)
p_values.sort_values(ascending=False, inplace=True)
p_values = pd.DataFrame(p_values, columns=['p_value'])
indpendientes = p_values[p_values['p_value'] > 0.05]
cat_col_to_drop = indpendientes.index.tolist()


df_final = df.drop(num_cols_to_drop, axis=1)
df_final = df_final.drop(cat_col_to_drop, axis=1)

file_path = root.DIR_DATA_STAGE + 'df_final_infouser.csv'
df_final.to_csv(file_path, index=False)

# info creditos

path_infocreditos = root.DIR_DATA_RAW + 'db_raw_creditinfo.csv'

df_infocreditos = pd.read_csv(path_infocreditos)

df_infocreditos['PRÓXIMA FECHA PAGO'] = pd.to_datetime(df_infocreditos['PRÓXIMA FECHA PAGO'])
df_infocreditos['mes'] = df_infocreditos['PRÓXIMA FECHA PAGO'].dt.month
df_infocreditos.drop(columns=['PRÓXIMA FECHA PAGO','INT CORRIENTE'], inplace=True)

file_path = root.DIR_DATA_STAGE + 'df_final_creditinfo.csv'
df_infocreditos.to_csv(file_path, index=False)







