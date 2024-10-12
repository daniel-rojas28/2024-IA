import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import  pandas as pd
# Cargar el dataset y preprocesar
# (Asegúrate de ajustar el path a tu dataset)
file_path = '../datasets/cirrhosis.csv'
df = pd.read_csv(file_path)

# Cambiar edad de días a años
df['Age'] = df['Age'] / 365
print(df.info())

# Codificar variables categóricas
label_encoders = {}
categorical = df.select_dtypes(include=['object']).columns
for col in categorical:
    le = LabelEncoder()
    label = le.fit_transform(df[col].astype(str))
    df[col] = label
    # Crear un diccionario con los labels y los valores originales
    label_encoders[col] = le

mappings = {}
for col, le in label_encoders.items():
    mappings[col] = {index: label for index, label in enumerate(le.classes_)}

print(mappings)
print(df.info())

# Imputar valores nulos con la mediana
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# Revisar valores nulos
print(df.isnull().sum())


# Separar variables predictoras y la variable objetivo (Stage)
X = df.drop(columns=['ID', 'Stage', 'Status', 'N_Days'])
y = df['Stage']

# Matriz de correlación
correlation_matrix = df.corr()
print(correlation_matrix)
# Diagrama de correlación
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14, 12))
plt.title('Matriz de Correlación de Cirrhosis Dataset', fontsize=16)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest con los datos sintéticos
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba originales
y_pred = rf_model.predict(X_test)

# Mostrar la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(classification_rep)

# Guardar el modelo
import joblib
joblib.dump(rf_model, '../models/cirrhosis.pkl')
