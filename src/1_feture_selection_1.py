# Import librerias de python
import os
import sys

# imprt librerias paar el procesamiento 
import pandas as pd

# cargar lso datos
current_dir = os.getcwd() # Obtener la ruta del directorio actual del notebook
ROOT_PATH = os.path.dirname(current_dir) # Obtener la ruta del directorio superior
sys.path.insert(1, ROOT_PATH) # Insertar la ruta en sys.path

import root # Importar el módulo root

df_path = root.DIR_DATA_RAW + 'db_v0.xlsx'
df_raw = pd.read_excel(df_path)
df_processed = df_raw.copy()

# eliminar registro sin informacion de mora o pago
df_processed = df_processed[df_processed['CAPITAL.1'] == 0]
# creacion de variables nuevas con ID_usuario
df_processed['NUM.CREDITOS SOLICITADOS'] = df_processed.groupby('ID_USUARIO').cumcount() + 1 
def usuario_recurrente(df):
    if df['NUM.CREDITOS SOLICITADOS'] > 1:
        return 1
    else:
        return 0
    
df_processed['USUARIO RECURRENTE'] = df_processed.apply(usuario_recurrente, axis=1)

# eliminacion columnas redundantes o inecesarias
columns_to_drop = ['ID_USUARIO', 'ID CREDITO', 'PRODUCTO', 'MONEDA', 'CUOTAS',
       'PERIODICIDAD CUOTAS','TASA CORRIENTE', 'TASA SEGURO',
       'TASA AVAL', 'IVA AVAL', 'DESC AVAL', 'DESC AVAL AL DESEMB','Cuotas pagadas',
       'GESTIÓN DIGITAL', 'DESC. X INCLUSION', 'IVA GEST DIG',
       'COD. PROMO DESC.', 'FACTURA VENTA','GESTIÓN DIGITAL.1', 'IVA',
       'VALOR DESEMBOLSADO', 'VALOR FUTURO', 'CAPITAL.1',
       'CAPITAL EN MORA', 'INT CORRIENTE.1', 'SEGURO.1', 'GESTIÓN DIGITAL.2',
       'IVA.1','GAC','Cuotas Futuras', 'ESTADO 2',
       'DEUDA A LA FECHA', 'DEUDA TOTAL CRÉDITO', 'SEGURO', 'INT MORA','ESTADO 1'
]

df_processed = df_processed.drop(columns_to_drop, axis=1)

# dividir el data set
df_users_info = df_processed.copy()
df_users_info = df_users_info.dropna(subset='AÑOS EN LA VIVIENDA')

users_info_columns = ['TIPO EMPLEO', 'CIUDAD RESIDENCIA', 'TRABAJO',
       'TIPO DE VIVIENDA', 'ESTRATO', 'AÑOS EN LA VIVIENDA',
       'INGRESOS MENSUALES', 'GASTOS MENSUALES', 'INGRESOS ADICIONALES',
       'TIPO DE CONTRATO', 'PERIODO DE PAGO', 'ESTADO CIVIL',
       'NIVEL EDUCATIVO', 'PERSONAS A CARGO', 'NUMERO DE HIJOS',
       'TIPO DE VEHICULO', 'TIEMPO TRABAJO']

df_credit_info = df_processed.drop(users_info_columns,axis=1)

# nueva varibakle target paar cretid info
df_credit_info.loc[:, 'mora'] = (df_credit_info['DÍAS MORA'] > 3).astype(int)

df_credit_info.drop(['DÍAS MORA','Cuotas en mora','FECHA DESEMBOLSO'], axis=1, inplace=True)

# guardar los datso en raw
ruta1 = root.DIR_DATA_RAW + 'db_raw_infousers.csv'
df_users_info.to_csv(ruta1, index=False) 

ruta2 = root.DIR_DATA_RAW + 'db_raw_creditinfo.csv'
df_credit_info.to_csv(ruta2, index=False) 