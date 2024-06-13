
# importar librerias de python
import os
import sys
# importar librerias para el procesamiento

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate , StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay , precision_recall_curve , RocCurveDisplay
from sklearn.metrics import accuracy_score,roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import joblib


current_dir = os.getcwd() # Obtener la ruta del directorio actual del notebook
ROOT_PATH = os.path.dirname(current_dir) # Obtener la ruta del directorio superior
sys.path.insert(1, ROOT_PATH) # Insertar la ruta en sys.path

import root # Importar el módulo root


# ## Modelos info users

file_path = root.DIR_DATA_STAGE + 'train_infousers.csv'


df_infousers = pd.read_csv(file_path)
df_infousers.head()


# ### info usuarios


X = df_infousers.drop(columns=['Cuotas en mora'])
X = StandardScaler().fit_transform(X)
y = df_infousers['Cuotas en mora']


# Inicializar los modelos
logistic_regression = LogisticRegression(max_iter=1000)
naive_bayes = GaussianNB()
lda = LinearDiscriminantAnalysis()


metricas = ['accuracy', 'precision', 'recall', 'f1','roc_auc','average_precision']


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


logistic_scores = cross_validate(logistic_regression, X, y, cv=cv, scoring=metricas)
naive_bayes_scores = cross_validate(naive_bayes, X, y, cv=cv, scoring=metricas)
lda_scores = cross_validate(lda, X, y, cv=cv, scoring=metricas)

def print_results(model_name, scores):
    print(f"\n{model_name}")
    for metrica in metricas:
        mean_score = np.mean(scores[f'test_{metrica}'])
        std_score = np.std(scores[f'test_{metrica}'])
        print(f"{metrica.capitalize()}: {mean_score:.2f} +/- {std_score * 2:.2f}")

print_results("Regresión Logística", logistic_scores)
print_results("Naive Bayes", naive_bayes_scores)
print_results("LDA", lda_scores)





# ## Modelos para CredInfo


file_path = root.DIR_DATA_STAGE + 'train_creditinfo.csv'


df_credit_info = pd.read_csv(file_path)
df_credit_info.head()


X = df_credit_info.drop(columns=['mora'])
X = StandardScaler().fit_transform(X)
y = df_credit_info['mora']


# ## Scores usando k startify K-fold


logistic_scores = cross_validate(logistic_regression, X, y, cv=cv, scoring=metricas)
naive_bayes_scores = cross_validate(naive_bayes, X, y, cv=cv, scoring=metricas)
lda_scores = cross_validate(lda, X, y, cv=cv, scoring=metricas)


print_results("Regresión Logística", logistic_scores)
print_results("Naive Bayes", naive_bayes_scores)
print_results("LDA", lda_scores)



# ## Modelación



def modelo(df,target_col,model):
    
    X = df.drop(columns=[target_col])
    X = StandardScaler().fit_transform(X)
    y = df[target_col]
    
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)
    
    model.fit(X, y)
    
    return model



file_path2 = root.DIR_DATA_STAGE + 'test_infousers.csv'
test_infousers = pd.read_csv(file_path2)



Modelo1 = modelo(df_infousers,'Cuotas en mora',logistic_regression)
model1_path = root.DIR_DATA_ANALYTICS + 'model1_infousers.pkl'
joblib.dump(Modelo1, model1_path)

file_path2 = root.DIR_DATA_STAGE + 'test_creditinfo.csv'
test_credit_info = pd.read_csv(file_path2)

Modelo2 = modelo(df_credit_info,'mora',logistic_regression)
model2_path = root.DIR_DATA_ANALYTICS + 'model2_creditinfo.pkl'
joblib.dump(Modelo2, model2_path)




