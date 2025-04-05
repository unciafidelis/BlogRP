En este post se presentan 5 diferentes implementaciones de modelos ML:

### 1. **Clasificación con el conjunto de datos Iris (Clasificación supervisada)**

#### Descripción:
El conjunto de datos Iris es uno de los más utilizados para tareas de clasificación. Contiene medidas de diferentes partes de flores de iris y tiene 3 clases de flores: Setosa, Versicolor, y Virginica. La tarea es predecir la especie de la flor en función de las medidas.

#### Implementación:

Primero, asegurémonos de tener las bibliotecas necesarias:

```bash
pip install scikit-learn matplotlib seaborn
```

```python
# Importar librerías necesarias
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de clasificación con RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualizar las predicciones vs reales
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt="d")
plt.title("Matriz de Confusión")
plt.show()
```

#### Explicación:
1. **Cargar el conjunto de datos Iris**: Utilizamos la función `load_iris()` de **scikit-learn** para cargar este conjunto de datos.
2. **División de datos**: Usamos `train_test_split()` para dividir los datos en conjuntos de entrenamiento (70%) y prueba (30%).
3. **Modelo de clasificación**: Utilizamos un clasificador RandomForest (`RandomForestClassifier`), que es muy efectivo para tareas de clasificación.
4. **Evaluación**: Después de entrenar el modelo, realizamos predicciones sobre el conjunto de prueba y calculamos la precisión con `accuracy_score()`. Finalmente, mostramos la matriz de confusión usando **seaborn**.

### 2. **Regresión Lineal con el conjunto de datos de Boston Housing (Regresión supervisada)**

#### Descripción:
El conjunto de datos de Boston Housing es otro clásico en la comunidad de Machine Learning. Contiene información sobre diversas características de casas en Boston (por ejemplo, tasa de criminalidad, nivel de ozono, etc.), y la tarea es predecir el valor medio de las casas en función de estas características.

#### Implementación:

```bash
pip install scikit-learn matplotlib seaborn
```

```python
# Importar librerías necesarias
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos de Boston Housing
boston = datasets.load_boston()
X = boston.data
y = boston.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualización de las predicciones vs reales
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Regresión Lineal: Valores reales vs predicciones")
plt.show()
```

#### Explicación:
1. **Cargar el conjunto de datos de Boston Housing**: Usamos `load_boston()` para cargar los datos. Este dataset tiene 13 características predictoras y un valor objetivo: el precio medio de la vivienda en miles de dólares.
2. **División de datos**: Separamos los datos en entrenamiento y prueba usando `train_test_split()`.
3. **Modelo de regresión lineal**: Usamos el modelo `LinearRegression` de **scikit-learn** para ajustar una línea recta a los datos.
4. **Evaluación**: Calculamos el error cuadrático medio (MSE) para evaluar la calidad del modelo, y visualizamos las predicciones en un gráfico de dispersión.

### 3. **Clasificación con el conjunto de datos de Digits (Clasificación supervisada)**

#### Descripción:
El conjunto de datos **Digits** es un conjunto de imágenes de dígitos escritos a mano (del 0 al 9). La tarea es clasificar estas imágenes en una de las 10 clases.

#### Implementación:

```bash
pip install scikit-learn matplotlib
```

```python
# Importar librerías necesarias
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Digits
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo SVM para clasificación
model = SVC(gamma=0.001)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualizar algunos dígitos con sus predicciones
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, i in zip(axes.ravel(), range(10)):
    ax.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray)
    ax.set_title(f"Pred: {y_pred[i]}")
    ax.axis("off")
plt.show()
```

#### Explicación:
1. **Cargar el conjunto de datos Digits**: Usamos `load_digits()` para obtener el conjunto de datos de imágenes de dígitos escritos a mano.
2. **División de datos**: Separamos los datos en entrenamiento y prueba con `train_test_split()`.
3. **Modelo de clasificación SVM**: Usamos una **Máquina de Vectores de Soporte** (SVM) para clasificar los dígitos. Configuramos `gamma=0.001` para controlar la complejidad del modelo.
4. **Evaluación**: Calculamos la precisión (`accuracy_score()`) y mostramos algunos ejemplos de los dígitos con sus predicciones.

### 4. **Clustering con el conjunto de datos de Iris (Clustering no supervisado)**

#### Descripción:
El **Clustering** es un tipo de aprendizaje no supervisado. El objetivo aquí es agrupar los datos sin etiquetas conocidas. Vamos a usar el algoritmo **K-Means** para agrupar las flores Iris en 3 grupos (como en la tarea de clasificación anterior).

#### Implementación:

```bash
pip install scikit-learn matplotlib seaborn
```

```python
# Importar librerías necesarias
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import seaborn as sns

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data

# Aplicar KMeans con 3 clusters (debido a las 3 especies de flores)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizar los clusters
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette="Set1", s=100)
plt.title("Clustering K-Means con el conjunto de datos Iris")
plt.xlabel("Longitud del sépalo")
plt.ylabel("Anchura del sépalo")
plt.show()
```

#### Explicación:
1. **Cargar el conjunto de datos Iris**: Cargamos el conjunto de datos Iris como en el primer ejemplo.
2. **Aplicar K-Means**: Usamos `KMeans` para agrupar los datos en 3 clusters, ya que sabemos que hay 3 especies diferentes.
3. **Visualización**: Usamos **Seaborn** para visualizar los clusters en un gráfico de dispersión.

---

### Conclusión

Estos son ejemplos básicos de implementación de modelos de Machine Learning en Python utilizando bibliotecas como **scikit-learn** y **matplotlib**. En resumen:

- **Clasificación** (Ejemplo 1 y 3): Predecir una clase a partir de características (por ejemplo, especies de flores o dígitos).
- **Regresión** (Ejemplo 2): Predecir un valor continuo (por ejemplo, precios de viviendas).
- **Clustering** (Ejemplo 4): Agrupar datos sin etiquetas utilizando técnicas no supervisadas.

Estos ejemplos te permiten entender conceptos básicos de ML como clasificación, regresión y clustering usando bases de datos públicas.
