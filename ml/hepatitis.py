import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../datasets/hepatitis.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)  # Eliminar la columna 'Unnamed: 0'
# Verificar la forma del DataFrame y las primeras filas
print(df.shape)  # Debe mostrar (615, 14)
print(df.head())  # Para verificar los datos

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Seleccionar solo columnas numéricas
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'm' else 0)  # Ejemplo para convertir 'Sex'

category_map = {
    "0=Blood Donor": 0,
    "1=Hepatitis": 1,
    "2=Fibrosis": 2,
    "3=Cirrhosis": 3,
    "0s=suspect Blood Donor": 4,
}
df['Category'] = df['Category'].map(category_map)

X = df.drop(columns=['Category'])
y = df['Category']

# Dividir los datos (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar las formas de los conjuntos
print(X_train.shape, X_test.shape)


# Definir los rangos de k
Ks = 20  # Número máximo de vecinos
mean_acc = []  # Para almacenar la precisión media
std_acc = []  # Para almacenar el desvío estándar de la precisión

# Entrenar el modelo para diferentes valores de k

for k in range(1, Ks):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5-fold cross-validation
    mean_acc.append(scores.mean())
    std_acc.append(scores.std())

# Graficar la precisión
plt.plot(range(1, Ks), mean_acc, 'g', label='Accuracy')
plt.fill_between(range(1, Ks), np.array(mean_acc) - 1 * np.array(std_acc),
                 np.array(mean_acc) + 1 * np.array(std_acc), alpha=0.05)
plt.legend(('Accuracy', '+/- 1 std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.title('k-NN Varying Number of Neighbors')
plt.tight_layout()
plt.show()

# Entrenar el modelo con el mejor valor de k
best_k = np.argmax(mean_acc) + 1
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Hacer predicciones
y_pred = knn.predict(X_test)

# Calcular la precisión
accuracy = np.mean(y_pred == y_test)

print(f"La precisión del modelo k-NN con k={best_k} es: {accuracy:.2f}")

# Guardar el modelo
import joblib

joblib.dump(knn, '../models/hepatitis.pkl')
print("Modelo guardado con éxito!")
