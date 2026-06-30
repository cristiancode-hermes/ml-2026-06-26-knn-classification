# 🎯 KNN Classification — ¿El vino es bueno? Predice la calidad del vino tinto

**Nivel:** 3 | **Skill:** Clasificación con KNN (K-Vecinos más Cercanos)

## 📖 Qué vas a aprender

### ¿Qué es clasificación?

Hasta ahora hemos visto **regresión** (predecir un número: precio, temperatura). La **clasificación** es distinta: predecimos una **categoría** (o clase).

Ejemplos del mundo real:
- ¿Este email es **spam** o **no spam**? → clasificación binaria (2 clases)
- ¿Este vino es de **baja**, **media** o **alta** calidad? → clasificación multiclase (3+ clases)
- ¿Esta imagen tiene un **gato**, un **perro** o un **pájaro**? → clasificación multiclase

### ¿Cómo funciona KNN?

KNN (K-Nearest Neighbors — K Vecinos más Cercanos) es uno de los algoritmos más intuitivos:

1. **Imagina** que tienes puntos en un mapa, cada punto tiene una etiqueta de color
2. Cuando llega un **punto nuevo** (sin etiqueta), miras los K vecinos más cercanos
3. El punto nuevo recibe la **etiqueta más común** entre sus K vecinos

![idea visual de KNN](docs/knn_accuracy.png)

### Parámetro K

K es el número de vecinos a considerar:
- **K pequeño** (ej: K=1): muy sensible al ruido, puede sobreajustar (overfitting)
- **K grande** (ej: K=20): más suave, pero puede perder detalles (underfitting)
- **Regla general**: K = √(n_muestras), probar varios valores y elegir el mejor

### ¿Por qué escalar las features?

KNN usa **distancias** (Euclidiana por defecto). Si una feature como `alcohol` (8-14) tiene valores mucho más grandes que `pH` (2.5-4), el alcohol domina la distancia. Escalamos para que todas las features pesen igual.

### Métricas de clasificación

- **Accuracy**: (correctas) / (totales) — simple, pero puede engañar si hay clases desbalanceadas
- **Precision**: de las que predije como "buenas", ¿cuántas eran realmente buenas?
- **Recall**: de las que eran realmente "buenas", ¿cuántas atrapé?
- **F1-score**: promedio armónico de precision y recall
- **Matriz de confusión**: tabla que muestra aciertos y errores por clase

```
              Predicho
            No     Sí
Real No    TN     FP    ← Falsos Positivos (falsa alarma)
     Sí    FN     TP    ← Falsos Negativos (lo dejamos pasar)
```

## 🧪 Código paso a paso

### Estructura del proyecto

```
2026-06-26-knn-classification/
├── README.md           ← Este archivo
├── project.py          ← Código principal (¡ejecutable!)
├── requirements.txt    ← Dependencias
├── data/
│   └── winequality-red.csv  ← Dataset real de vinos tintos
├── docs/               ← Visualizaciones generadas (PNG)
│   ├── 01_calidad_distribucion.png
│   ├── 02_correlaciones.png
│   ├── 03_k_optimizacion.png
│   ├── 04_matriz_confusion.png
│   └── 05_decision_boundary.png
├── models/             ← Modelos guardados (cuando toque)
└── LEARNINGS.md        ← Qué aprendiste hoy
```

### Paso 1: Cargar y explorar el dataset

Primero cargamos los datos con pandas. El archivo usa punto y coma (;) como separador.

```python
import pandas as pd
df = pd.read_csv('data/winequality-red.csv', sep=';')
```

### Paso 2: Crear la variable objetivo

La columna `quality` va del 3 al 8. Vamos a convertir esto en un problema de clasificación binaria: **vino bueno** (quality >= 7) vs **vino normal** (quality < 7).

```python
df['good_quality'] = (df['quality'] >= 7).astype(int)
```

Esto nos da ~87% de vinos "normales" y ~13% "buenos" — un desbalanceo realista.

### Paso 3: KNN paso a paso

1. Separamos features (X) y target (y)
2. Dividimos en train (80%) y test (20%)
3. Escalamos features con StandardScaler
4. Entrenamos KNN con K=5
5. Evaluamos con accuracy, matriz de confusión, y más

### Paso 4: Encontrar el mejor K

Probamos K = 1 hasta K = 50 y vemos cuál da mejor accuracy:

```python
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))
```

### Paso 5: Visualizar la frontera de decisión

Usamos PCA para reducir las 11 dimensiones a 2 y visualizar cómo KNN separa las clases:

- Train points → colores según su clase real
- Fondo coloreado → región que KNN asigna a cada clase
- La frontera es la línea imaginaria donde KNN cambia de opinión

## 📊 Resultados

Al ejecutar `python project.py` verás:

1. **Distribución de calidad**: histograma mostrando cuántos vinos hay de cada calidad
2. **Correlaciones**: mapa de calor mostrando qué variables se relacionan con la calidad
3. **Accuracy del modelo**: ~75-80% en test
4. **Mejor K**: gráfico mostrando accuracy vs K (con el óptimo marcado)
5. **Matriz de confusión**: cuántos vinos clasificó bien y mal
6. **Frontera de decisión**: visualización 2D de cómo KNN separa las clases

### Output esperado (aproximado)

```
=== 🍷 CLASIFICADOR DE VINOS CON KNN ===

📊 Datos cargados: 1599 vinos, 12 columnas
   Features: fixed acidity, volatile acidity, citric acid, ...

🎯 Clases: 0=Normal (86.6%), 1=Buena (13.4%)

📈 Accuracy en entrenamiento: 82.3%
📈 Accuracy en test:        76.8%

📊 Mejor K encontrado: K=13 con accuracy=78.2%

📋 Reporte de clasificación:
              precision    recall  f1-score
Normal          0.90       0.83      0.86
Buena           0.41       0.56      0.47
```

## 💡 Conceptos clave

- **Clasificación**: predecir una categoría, no un número
- **KNN**: algoritmo basado en distancia que vota por mayoría entre los K vecinos
- **Importancia del escalado**: KNN requiere features en la misma escala
- **Hiperparámetro K**: controla el balance entre sesgo y varianza
- **Matriz de confusión**: herramienta fundamental para evaluar clasificadores
- **Precision vs Recall**: dos formas distintas de medir errores
- **Desbalanceo de clases**: cuando una clase es mucho más común que la otra

## 🚀 Siguiente paso

Después de KNN, el siguiente proyecto será **Árboles de Decisión** — otro algoritmo de clasificación que es más interpretable (puedes ver las reglas que aprende). Luego combinaremos ambos conceptos en **Random Forest**, el ensamble de árboles que es uno de los algoritmos más potentes del ML clásico.
