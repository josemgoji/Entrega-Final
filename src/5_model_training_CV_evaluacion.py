
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


# ### Naive Bayes


##Función pára curva lift

def plot_comulative_gain(y_true, y_proba, ax):
    # Create a dataframe with true labels and predicted probabilities
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    
    # Calculate the cumulative gain for the lift curve
    df['cumulative_data_fraction'] = (np.arange(len(df)) + 1) / len(df)
    df['cumulative_positive_fraction'] = np.cumsum(df['y_true']) / df['y_true'].sum()
    
    # Plot the lift curve
    ax.plot(df['cumulative_data_fraction'], df['cumulative_positive_fraction'], label='Cumulative Gain Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline')
    ax.set_title('Cumulative Gain Curve')
    ax.set_xlabel('Cumulative Data Fraction')
    ax.set_ylabel('Cumulative Positive Fraction')
    ax.legend()


def plot_lift_curve(y_true, y_proba, ax):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    
    # Calculate the cumulative gain for the lift curve
    df['cumulative_data_fraction'] = (np.arange(len(df)) + 1) / len(df)
    df['cumulative_positive_fraction'] = np.cumsum(df['y_true']) / df['y_true'].sum()
    df['lift'] = df['cumulative_positive_fraction'] / df['cumulative_data_fraction']
    
    ax.plot(df['cumulative_data_fraction'], df['lift'], label='Lift Curve')
    ax.set_title('Lift Curve')
    ax.set_xlabel('Cumulative Data Fraction')
    ax.set_ylabel('Lift')
    ax.legend()
    


def plot_precision_recall_curve(y, y_proba, ax):
    ax.set_title('Train Data')
    prsn, rcll, _ = precision_recall_curve(y, y_proba)
    AP = average_precision_score(y, y_proba)
    ax.plot(rcll, prsn, label= f'Precison-Recall AP = {round(AP,2)})')
    ax.set_title('Precision Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()


def modelo(df,target_col,model):
    
    X = df.drop(columns=[target_col])
    X = StandardScaler().fit_transform(X)
    y = df[target_col]
    
    ros = RandomOverSampler(random_state=0)
    X, y = ros.fit_resample(X, y)
    
    model.fit(X, y)
    
    return model


def graficas(df, target_col, model):
    
    scaler = StandardScaler()
    X = df.drop(columns=[target_col])
    columns = X.columns
    X = scaler.fit_transform(X)
    y = df[target_col]
   
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] 
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    AP = average_precision_score(y, y_proba)
    AUC = roc_auc_score(y, y_proba)

   
    print("Accuracy: {:.3f}".format(accuracy))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1 Score: {:.3f}".format(f1))
    print("Average Precision: {:.3f}".format(AP))
    print("Area Under the ROC Curve: {:.3f}".format(AUC))
    print('------------------------------------------------------------------------------')
    #print(classification_report(y, y_pred))
    
    
    fig1, axs = plt.subplots(1, 1, figsize=(10, 6))
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y, normalize='true', xticks_rotation='vertical', ax=axs)
    axs.set_title("Confusion Matrix")  # Add a title to the plot
    plt.show()
    

    fig2, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    disp = RocCurveDisplay.from_estimator(model, X, y, ax=axs[0][0])
    axs[0][0].set_title('ROC Curve')
    axs[0][0].plot([0, 1], [0, 1], linestyle='--', color='grey', label='Baseline')
    plot_precision_recall_curve(y, y_proba, ax=axs[0][1])
    plot_lift_curve(y, y_proba, ax=axs[1][0])
    plot_comulative_gain(y, y_proba, ax=axs[1][1])
    
    fig2.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.show()
    
    df = pd.DataFrame(scaler.inverse_transform(X), columns=columns)
    
    df['Estado'] = y
    df['Estado_Estimado'] = y_pred
    df['Probabilidad'] = np.round(y_proba * 100,2)
    
    return df, fig1, fig2



file_path2 = root.DIR_DATA_STAGE + 'test_infousers.csv'
test_infousers = pd.read_csv(file_path2)



Modelo1 = modelo(df_infousers,'Cuotas en mora',logistic_regression)
df_final, fig1_1, fig2_1 = graficas(test_infousers, 'Cuotas en mora', model = Modelo1)

file_path2 = root.DIR_DATA_STAGE + 'test_creditinfo.csv'
test_credit_info = pd.read_csv(file_path2)

Modelo2 = modelo(df_credit_info,'mora',logistic_regression)
df_final_2, fig1_2, fig2_2 = graficas(test_credit_info, 'mora', model = Modelo1)


## guardar graficas, modelos y df final
df_final_path = root.DIR_DATA_ANALYTICS + 'df_final_infousers.csv'
cm_path = root.DIR_DATA_ANALYTICS + 'confusion_matrix_infousers.png'
plots_path = root.DIR_DATA_ANALYTICS + 'graficas_infousers.png'
model1_path = root.DIR_DATA_ANALYTICS + 'model1_infousers.pkl'
fig1_1.savefig(cm_path)
fig2_1.savefig(plots_path)
df_final.to_csv(df_final_path, index=False)
joblib.dump(Modelo1, model1_path)

df_final2_path = root.DIR_DATA_ANALYTICS + 'df_final_creditinfo.csv'
cm2_path = root.DIR_DATA_ANALYTICS + 'confusion_matrix_creditinfo.png'
plots2_path = root.DIR_DATA_ANALYTICS + 'graficas_creditinfo.png'
model2_path = root.DIR_DATA_ANALYTICS + 'model2_creditinfo.pkl'
fig1_2.savefig(cm2_path)
fig2_2.savefig(plots2_path)
df_final_2.to_csv(df_final2_path, index=False)
joblib.dump(Modelo2, model2_path)



