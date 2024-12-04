import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import plot_tree

# 1. Análisis Exploratorio de Datos (EDA)
print("1. Análisis Exploratorio de Datos")

# Cargar los datos
df = pd.read_csv('https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Titanic-Dataset-ARN6IbjPT4VRYsDHxVBxA6sn9kZJH3.csv')

# Mostrar información básica sobre el conjunto de datos
print(df.info())

# Mostrar estadísticas resumidas
print(df.describe())

# Verificar valores faltantes
print(df.isnull().sum())

# Visualizar la distribución de la variable objetivo (Survived)
plt.figure(figsize=(8, 6))
df['Survived'].value_counts().plot(kind='bar')
plt.title('Distribución de Supervivencia')
plt.xlabel('Sobrevivió')
plt.ylabel('Cantidad')
plt.savefig('distribucion_supervivencia.png')
plt.close()

# Visualizar la relación entre Pclass y Survival
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Supervivencia por Clase de Pasajero')
plt.savefig('supervivencia_por_clase.png')
plt.close()

# Visualizar la relación entre Sex y Survival
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Supervivencia por Género')
plt.savefig('supervivencia_por_genero.png')
plt.close()

# Visualizar la distribución de Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Distribución de Edad')
plt.savefig('distribucion_edad.png')
plt.close()

# Mapa de calor de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación')
plt.savefig('mapa_calor_correlacion.png')
plt.close()

print("\n2. Preprocesamiento de Datos")

# Manejar valores faltantes
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)  # Eliminar Cabin debido a la alta cantidad de valores faltantes

# Codificar variables categóricas
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Crear variables dummy para Pclass
df = pd.get_dummies(df, columns=['Pclass'], prefix='Class')

# Eliminar columnas innecesarias
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print(df.head())

print("\n3. Selección de Características")

X = df.drop('Survived', axis=1)
y = df['Survived']

# Seleccionar las 5 mejores características
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Obtener los nombres de las características seleccionadas
selected_features = X.columns[selector.get_support()].tolist()
print("Características seleccionadas:", selected_features)

X = X[selected_features]

print("\n4. División del Conjunto de Datos")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n5. Entrenamiento del Modelo de Árbol de Decisión")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("\n6. Evaluación del Modelo")

y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

print("\n7. Visualización de Resultados")

plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=selected_features, class_names=['No Sobrevivió', 'Sobrevivió'], filled=True, rounded=True)
plt.title("Visualización del Árbol de Decisión")
plt.savefig('arbol_decision.png')
plt.close()

# Importancia de las características
importances = dt.feature_importances_
feature_importance = pd.DataFrame({'caracteristica': selected_features, 'importancia': importances})
feature_importance = feature_importance.sort_values('importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importancia', y='caracteristica', data=feature_importance)
plt.title('Importancia de las Características')
plt.savefig('importancia_caracteristicas.png')
plt.close()

print("\n8. Interpretación y Análisis de Resultados")
print("Ver el análisis detallado en el informe adjunto.")

