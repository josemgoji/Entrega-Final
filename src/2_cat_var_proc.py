# import librerias de python
import os
import sys

# imprt librerias para el procesamiento
import pandas as pd
import re
from unidecode import unidecode
from fuzzywuzzy import fuzz
import numpy as np
import locale

# cargar los datos
current_dir = os.getcwd() # Obtener la ruta del directorio actual del notebook
ROOT_PATH = os.path.dirname(current_dir) # Obtener la ruta del directorio superior
sys.path.insert(1, ROOT_PATH) # Insertar la ruta en sys.path

import root # Importar el módulo root

df_path = root.DIR_DATA_RAW + 'db_raw_infousers.csv'

df = pd.read_csv(df_path)

#manejo de fechas
df['FECHA DESEMBOLSO'] = df['FECHA DESEMBOLSO'].str.replace('/', '-')
df['FECHA DESEMBOLSO'] = pd.to_datetime(df['FECHA DESEMBOLSO'], errors='coerce', format='%m-%d-%Y')

df['PRÓXIMA FECHA PAGO'] = df['PRÓXIMA FECHA PAGO'].str.replace('/', '-')
df['PRÓXIMA FECHA PAGO'] = pd.to_datetime(df['PRÓXIMA FECHA PAGO'], errors='coerce', format='%m-%d-%Y')

# Procesamiento de texto
def processing_text(texto):
    if texto == np.nan:
        return np.nan
    texto = str(texto)
    ### Limpieza
    texto = texto.lower() # Estandarizar todo a minúscula
    texto = unidecode(texto)  # Eliminar tildes
    texto = re.sub(r'[^a-zA-Z0-9\s/-]', '', texto)
    texto = re.sub(r'\s+', ' ', texto) # Eliminar espacios en blanco adicionales
    texto = texto.strip() # Eliminar espacios al inicio y al final
    
    return texto  

categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].map(processing_text)

# elimibo las ciudades que tienen tambien departamentos
departamentos_colombia = [
    "amazonas",
    "antioquia",
    "arauca",
    "atlantico",
    "bolivar",
    "boyaca",
    "caldas",
    "caqueta",
    "casanare",
    "cauca",
    "cesar",
    "choco",
    "cordoba",
    "cundinamarca",
    "guainia",
    "guaviare",
    "huila",
    "la guajira",
    "magdalena",
    "meta",
    "narino",
    "norte de santander",
    "putumayo",
    "quindio",
    "risaralda",
    "san andres y providencia",
    "santander",
    "sucre",
    "tolima",
    "valle", "cauca",
    "vaupes",
    "vichada"
]

def mapear_ciudades(df):
    mapeo = {}

    for ciudad in df['CIUDAD RESIDENCIA']:
        if isinstance(ciudad, str):  # Verificar si es un string
            ciudad_estandar = None
            # Buscar si la ciudad ya está en el diccionario
            for ciudad_mapeada in mapeo:
                if fuzz.ratio(ciudad.lower(), ciudad_mapeada.lower()) > 80:  # Umbral de similitud del 80%
                    ciudad_estandar = ciudad_mapeada
                    break
            if ciudad_estandar:
                mapeo[ciudad_estandar].append(ciudad)
            else:
                mapeo[ciudad] = [ciudad]

    return mapeo

def obtener_ciudad_estandar(ciudad, mapeo):
    if pd.isna(ciudad):
        return np.nan
    if isinstance(ciudad, str):  # Verificar si es un string
        for ciudad_estandar, variantes in mapeo.items():
            if ciudad in variantes:
                return ciudad_estandar
    return ciudad

# Crear el mapeo
mapeo = mapear_ciudades(df)

# Crear una nueva columna en el DataFrame con la ciudad estandarizada
df['ciudad_estandarizada'] = df['CIUDAD RESIDENCIA'].apply(lambda x: obtener_ciudad_estandar(x, mapeo))

# cargar df de ciudades de colombia
cidades_path = root.DIR_DATA_RAW + 'ciudades.csv'
df_ciudades = pd.read_csv(cidades_path)


index_to_drop = df_ciudades[df_ciudades['name'] == 'caramanta'].index
df_ciudades = df_ciudades.drop(index_to_drop)

## estandarizar las ciudades

def check_similarity_and_estandarize(x, cities_list):
    if not isinstance(x, str) or pd.isnull(x):
        return np.nan
    for city in cities_list:
        similarity = fuzz.ratio(x, city)
        if similarity >= 80:
            return city
    return np.nan

def transform_string(x):
    cities = ['girardota','copacabana']
    for city in cities:
        if x == city:
            return 'bello'
    return x

df['ciudad_estandarizada'] = df['ciudad_estandarizada'].apply(transform_string)

df['ciudad_estandarizada'] = df['CIUDAD RESIDENCIA'].apply(check_similarity_and_estandarize, cities_list=df_ciudades['name'].values)
df['ciudad_estandarizada'].fillna(df['ciudad_estandarizada'].mode()[0], inplace=True)

# columna de ubicacion longitud y latitud
city_coords = df_ciudades.set_index('name')['location'].T.to_dict()

df['Ubicacion'] = df['ciudad_estandarizada'].map(city_coords)

regex = r'"latitude":(-?\d+\.\d+),"longitude":(-?\d+\.\d+)'
def extraer_latitud(ubicacion):
    if pd.isna(ubicacion):
        return None
    match = re.search(regex, ubicacion)
    if match:
        return float(match.group(1))
    else:
        return None

# Función para extraer longitud
def extraer_longitud(ubicacion):
    if pd.isna(ubicacion):
        return None
    match = re.search(regex, ubicacion)
    if match:
        return float(match.group(2))
    else:
        return None

# Crear columnas de latitud y longitud
df['latitud'] = df['Ubicacion'].apply(extraer_latitud)
df['longitud'] = df['Ubicacion'].apply(extraer_longitud)

# procesamiento variabel trabajo
def eliminar_valores_numericos_grandes(valor):
    if str(valor).isnumeric() and len(str(valor)) > 2:
        return np.nan
    return valor

df['TIEMPO TRABAJO'] = df['TIEMPO TRABAJO'].apply(eliminar_valores_numericos_grandes)

def agregar_anios(valor):
    if str(valor).isnumeric():
        return str(valor) + ' anos'
    return valor

df['TIEMPO TRABAJO'] = df['TIEMPO TRABAJO'].apply(agregar_anios)

# Separar datos con letras y números
df['TIEMPO TRABAJO'] = df['TIEMPO TRABAJO'].fillna('nan')
letras_numeros = df['TIEMPO TRABAJO'].str.contains(r'[a-zA-Z]', regex=True)
#letras_numeros = letras_numeros.fillna('nan')  # Rellenar NaN con False
df['letras_numeros'] = df['TIEMPO TRABAJO'][letras_numeros]
df['letras_numeros'] = df['letras_numeros'].replace('nan', np.nan)
df['fechas_trab'] = df['TIEMPO TRABAJO'][~letras_numeros]

def es_fecha(texto):
    if pd.isna(texto):
        return np.nan
    texto = re.sub(r'(\D)(\d)', r'\1 \2', texto)
    texto = re.sub(r'[^a-zA-Z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    
    meses = r'(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)'
    patron = rf'({meses} \d{{1,2}} de \d{{4}}|{meses} \d{{1,2}} \d{{4}}|{meses} – \d{{1,2}} – \d{{4}}|{meses}\d{{1,2}}-\d{{4}})'
    match = re.search(patron, texto)
    if match:
        return match.group()

df['fechas'] = df['letras_numeros'].apply(es_fecha)

def obtener_valor_numerico(texto):
    if texto is not np.nan:
        texto = str(texto)
        texto = texto.lower()
        meses = 0
        anios = 0
        if 'mes' in texto:
            meses = re.findall(r'(\d+)\s*(?:mes?|meses?)', texto)
            if len(meses)==0:
                meses = 1
            else:    
                meses = int(meses[0])
                
        if any(word in texto.lower() for word in ['años', 'anos', 'año', 'años', 'ano']):
            anios = re.findall(r'(\d+)\s*(?:años?|anos?|año)', texto)
            if len(anios)==0:
                anios = 12
            else:    
                anios = 12*int(anios[0])
        if 'medio' in texto:
            meses = meses + 6
        return meses + anios
    return np.nan

df['letras_numeros'] = df['letras_numeros'].apply(obtener_valor_numerico)

locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

df['fechas'] = df['fechas'].str.replace(' de ', ' ').str.capitalize()
df['fechas'] = pd.to_datetime(df['fechas'], format='%B %d %Y') 

df['fechas_trab'] = df['fechas_trab'].str.replace('/', '-')
df['TIEMPO TRABAJO FORMATO 1'] = pd.to_datetime(df['fechas_trab'], format='%Y-%m-%d', errors='coerce')
df['TIEMPO TRABAJO FORMATO 2'] = pd.to_datetime(df['fechas_trab'], format='%d-%m-%Y', errors='coerce')
df['TIEMPO TRABAJO FORMATO 2'] = df['TIEMPO TRABAJO FORMATO 2'].dt.strftime('%Y-%m-%d')
df['fechas_trab'] = df['TIEMPO TRABAJO FORMATO 1'].fillna(df['TIEMPO TRABAJO FORMATO 2'])
df['fechas_trab'] = df['fechas_trab'].fillna(df['fechas'])

df['meses_transcurridos'] = ((df['FECHA DESEMBOLSO'] - df['fechas_trab']) / pd.Timedelta(days=30.44))

def convertir_a_nan(valor):
    if pd.isnull(valor) or valor < 0:
        return np.nan
    return valor

df['meses_transcurridos'] = df['meses_transcurridos'].apply(convertir_a_nan)

df['meses_transcurridos'] = df['meses_transcurridos'].fillna(df['letras_numeros'])

## Columnas a eliminar
columas_to_drop = ['TIEMPO TRABAJO','fechas','fechas_trab','TIEMPO TRABAJO FORMATO 1',
                   'TIEMPO TRABAJO FORMATO 2','letras_numeros', 'FECHA DESEMBOLSO',
                   'CIUDAD RESIDENCIA','Ubicacion','TRABAJO','ciudad_estandarizada']
df.drop(columas_to_drop, axis=1, inplace=True)

#guardar los datos

ruta1 = root.DIR_DATA_STAGE + 'db_pross_info_users.csv'
df.to_csv(ruta1, index=False)
print('guardado') 