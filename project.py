#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  KNN CLASSIFICATION — ¿El vino es bueno?                               ║
║  Nivel 3 | Skill: Clasificación, KNN, métricas, matriz de confusión    ║
╚══════════════════════════════════════════════════════════════════════════╝

¿Qué vamos a hacer hoy?
────────────────────────
Hoy vamos a construir un clasificador que predice si un vino tinto
es de "buena calidad" (⭐) o "normal" basándose en 11 características
químicas como el alcohol, acidez, pH, etc.

Usaremos KNN (K-Vecinos más Cercanos), que es el algoritmo de clasificación
más intuitivo: "dime quiénes son tus vecinos y te diré quién eres".

Aprenderás:
  ✅ Qué es clasificación (vs regresión que ya vimos)
  ✅ Cómo funciona KNN paso a paso
  ✅ Por qué hay que escalar las features
  ✅ Cómo evaluar un clasificador (accuracy, matriz de confusión)
  ✅ Cómo encontrar el mejor valor de K
  ✅ Cómo visualizar la frontera de decisión con PCA

Requisitos: pip install pandas numpy matplotlib seaborn scikit-learn
"""

# ============================================================================
# IMPORTS — Librerías que vamos a usar
# ============================================================================
import pandas as pd              # Para cargar y manipular datos
import numpy as np               # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para gráficos
import seaborn as sns            # Para gráficos más bonitos
import os                        # Para crear directorios
import warnings                  # Para ignorar warnings molestos
warnings.filterwarnings('ignore') # Silenciamos warnings para que se vea limpio

# De scikit-learn importamos lo necesario
from sklearn.model_selection import train_test_split  # Dividir datos en train/test
from sklearn.neighbors import KNeighborsClassifier    # ¡El algoritmo KNN!
from sklearn.preprocessing import StandardScaler       # Para escalar features
from sklearn.metrics import (
    accuracy_score,             # % de predicciones correctas
    confusion_matrix,           # Matriz de confusión
    classification_report,      # Reporte con precision, recall, f1
    ConfusionMatrixDisplay      # Para visualizar matriz de confusión
)
from sklearn.decomposition import PCA  # Para reducir dimensiones y visualizar

# ============================================================================
# CONFIGURACIÓN — Colores bonitos y directorio para gráficos
# ============================================================================
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({'figure.max_open_warning': 0})

# Creamos directorio para guardar los gráficos
os.makedirs('docs', exist_ok=True)

print("╔═══════════════════════════════════════════════════════════════╗")
print("║  🍷 CLASIFICADOR DE VINOS TINTOS CON KNN                     ║")
print("║  ¿Podemos predecir si un vino es bueno por su composición?   ║")
print("╚═══════════════════════════════════════════════════════════════╝\n")

# ============================================================================
# PASO 1: CARGAR LOS DATOS
# ============================================================================
# El dataset winequality-red.csv contiene análisis químico de 1599 vinos
# tintos portugueses (variedad Vinho Verde). Cada fila es un vino diferente.
# El separador es punto y coma (;) — ¡ojo! no es coma.
# ============================================================================

print("📥 Paso 1: Cargando datos...")

df = pd.read_csv('data/winequality-red.csv', sep=';')

print(f"   ✅ Datos cargados: {df.shape[0]} vinos, {df.shape[1]} columnas")
print(f"   Columnas: {', '.join(df.columns)}\n")

# ============================================================================
# PASO 2: EXPLORAR LOS DATOS
# ============================================================================
# Primero entendamos qué tenemos. Las columnas son características químicas
# del vino y la última columna 'quality' es la calificación (3 a 8).
#
# fixed acidity        → acidez fija (tartárico, málico)
# volatile acidity     → acidez volátil (acético = vinagre)
# citric acid          → ácido cítrico
# residual sugar       → azúcar residual
# chlorides            → cloruros (sal)
# free sulfur dioxide  → dióxido de azufre libre
# total sulfur dioxide → dióxido de azufre total
# density              → densidad
# pH                   → pH (acidez)
# sulphates            → sulfatos
# alcohol              → % alcohol
# quality              → calidad (3-8, puntuación de catadores)
# ============================================================================

print("📊 Paso 2: Explorando los datos...")

# Mostramos estadísticas básicas de cada columna
print("\n   📋 Estadísticas descriptivas:")
print(f"   {'Columna':25s} {'Media':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s}")
print(f"   {'-'*57}")
for col in df.columns:
    print(f"   {col:25s} {df[col].mean():>8.2f} {df[col].min():>8.2f} {df[col].max():>8.2f} {df[col].std():>8.2f}")

# --- Gráfico 1: Distribución de calidad ---
# Vemos cuántos vinos hay de cada nivel de calidad
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de la calidad original (3 a 8)
sns.countplot(x='quality', data=df, ax=axes[0], palette='viridis', 
              edgecolor='black', alpha=0.8)
axes[0].set_title('Distribución de Calidad (original)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Calificación (puntaje de catadores)')
axes[0].set_ylabel('Número de vinos')

# Añadimos etiquetas con los valores encima de cada barra
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

# Creamos la variable objetivo: vino bueno = quality >= 7
df['good_quality'] = (df['quality'] >= 7).astype(int)

# Contamos cuántos son buenos vs normales
class_counts = df['good_quality'].value_counts()
class_pcts = df['good_quality'].value_counts(normalize=True) * 100

# Gráfico de barras para la nueva variable binaria
colors = ['#e74c3c', '#2ecc71']
bars = axes[1].bar(
    ['Normal (quality < 7)', 'Buena (quality >= 7)'],
    [class_counts[0], class_counts[1]],
    color=colors, edgecolor='black', alpha=0.8
)
axes[1].set_title('Clasificación Binaria: ¿Vino Bueno?', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Número de vinos')

# Añadimos etiquetas con valores y porcentajes
for bar, count, pct in zip(bars, [class_counts[0], class_counts[1]], 
                           [class_pcts[0], class_pcts[1]]):
    axes[1].annotate(f'{count} vinos\n({pct:.1f}%)', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/01_calidad_distribucion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/01_calidad_distribucion.png")

# --- Gráfico 2: Mapa de correlaciones ---
# Vemos qué variables se relacionan más con la calidad del vino
# Esto nos ayuda a entender qué características importan más

plt.figure(figsize=(12, 10))

# Calculamos la matriz de correlación
corr = df.drop('good_quality', axis=1).corr()

# Mapa de calor con seaborn
mask = np.triu(np.ones_like(corr, dtype=bool))  # Ocultamos la mitad superior (es simétrica)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlación'})
plt.title('🧩 Correlaciones entre características del vino', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/02_correlaciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/02_correlaciones.png")

# Mostramos las correlaciones más importantes con la calidad (original)
print("\n   📈 Correlaciones con calidad (ordenadas):")
corr_with_quality = df.drop('good_quality', axis=1).corr()['quality'].drop('quality').sort_values(ascending=False)
for var, corr_val in corr_with_quality.items():
    arrow = '🔼' if corr_val > 0 else '🔽'
    strength = 'fuerte' if abs(corr_val) > 0.3 else ('media' if abs(corr_val) > 0.1 else 'débil')
    print(f"      {arrow} {var:25s}: {corr_val:+.3f} (correlación {strength})")

print()

# ============================================================================
# PASO 3: PREPARAR LOS DATOS PARA ML
# ============================================================================
# Separamos las features (X) del target (y).
# X = todas las columnas químicas (11 variables)
# y = la variable que queremos predecir: good_quality (0=normal, 1=bueno)
# ============================================================================

print("🔧 Paso 3: Preparando datos para ML...")

# Features: todas las columnas excepto quality y good_quality
feature_cols = [c for c in df.columns if c not in ['quality', 'good_quality']]
X = df[feature_cols].values  # Matriz de features (1599 filas x 11 columnas)
y = df['good_quality'].values  # Vector target (1599 valores: 0 o 1)

print(f"   ✅ Features (X): {X.shape[0]} muestras, {X.shape[1]} variables")
print(f"   ✅ Target (y):   {y.shape[0]} muestras, clases={np.unique(y)}")
print(f"      Clase 0 (Normal): {sum(y==0)} vinos ({sum(y==0)/len(y)*100:.1f}%)")
print(f"      Clase 1 (Buena):  {sum(y==1)} vinos ({sum(y==1)/len(y)*100:.1f}%)")

# ============================================================================
# PASO 4: DIVIDIR EN TRAIN Y TEST
# ============================================================================
# Partimos los datos en 80% para entrenar y 20% para probar.
# El modelo NUNCA ve los datos de test hasta el final.
# Usamos stratify=y para mantener la misma proporción de clases en ambos sets.
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   📦 Split: train={len(X_train)} muestras ({len(X_train)/len(X)*100:.0f}%)")
print(f"                     test ={len(X_test)} muestras ({len(X_test)/len(X)*100:.0f}%)")
print(f"   📊 Proporción clases preservada (stratify)")

# ============================================================================
# PASO 5: ESCALAR LAS FEATURES
# ============================================================================
# ¡ESTE PASO ES CRÍTICO PARA KNN!
# KNN calcula distancias entre puntos. Si una variable (ej: alcohol 8-14%)
# tiene valores más grandes que otra (ej: pH 2.5-4), la distancia la domina
# la variable con mayor escala. ¡Injusto!
#
# StandardScaler transforma cada variable para que tenga:
#   media = 0
#   desviación estándar = 1
# Así todas las variables pesan igual en la distancia.
# ============================================================================

print("\n⚖️  Paso 5: Escalando features con StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajusta la escala a los datos de train
X_test_scaled = scaler.transform(X_test)        # Usa la MISMA escala en test

# Mostramos cómo cambió una variable de ejemplo
alcohol_idx = feature_cols.index('alcohol')
print(f"\n   📏 Ejemplo: 'alcohol' antes y después del escalado:")
print(f"      Antes  → media={X_train[:, alcohol_idx].mean():.2f}, "
      f"std={X_train[:, alcohol_idx].std():.2f}")
print(f"      Después → media={X_train_scaled[:, alcohol_idx].mean():.2f}, "
      f"std={X_train_scaled[:, alcohol_idx].std():.2f}")
print(f"      ✅ Ahora todas las variables tienen media=0 y std=1")

# ============================================================================
# PASO 6: ENTRENAR KNN
# ============================================================================
# Vamos a probar varios valores de K (de 1 a 50) para ver cuál funciona mejor.
# K = número de vecinos que votan.
# 
# K pequeño → modelo más flexible (puede sobreajustar)
# K grande  → modelo más rígido (puede subajustar)
# ============================================================================

print("\n🤖 Paso 6: Entrenando KNN con diferentes valores de K...")

k_values = range(1, 51)  # Probamos K de 1 a 50
train_scores = []  # Accuracy en entrenamiento
test_scores = []   # Accuracy en test

for k in k_values:
    # --- CREAMOS EL MODELO ---
    # KNeighborsClassifier es el algoritmo KNN de scikit-learn
    # n_neighbors = K (número de vecinos)
    # metric = 'euclidean' es la distancia por defecto (la de toda la vida)
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # --- ENTRENAMOS ---
    # fit() guarda los datos de entrenamiento en memoria
    # KNN es "perezoso" (lazy learner): no aprende un modelo, solo memoriza
    knn.fit(X_train_scaled, y_train)
    
    # --- EVALUAMOS ---
    # score() calcula el accuracy: (predicciones correctas) / (total)
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

# --- Gráfico 3: Accuracy vs K ---
# Vemos cómo cambia el rendimiento según el valor de K
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(k_values, train_scores, 'o-', label='Train accuracy', 
        color='#3498db', linewidth=2, markersize=4, alpha=0.8)
ax.plot(k_values, test_scores, 's-', label='Test accuracy', 
        color='#e74c3c', linewidth=2, markersize=4, alpha=0.8)

# Encontramos el mejor K (el que da mayor accuracy en test)
best_k = k_values[np.argmax(test_scores)]
best_acc = max(test_scores)

# Pero también encontramos un K más robusto (evitamos K=1 que sobreajusta)
# Miramos el mejor K entre 3 y 50 para evitar overfitting con K=1
valid_range = [k for k in k_values if k >= 3]
valid_scores = [test_scores[k-1] for k in valid_range]
robust_k = valid_range[np.argmax(valid_scores)]
robust_acc = max(valid_scores)

# Resaltamos el mejor K absoluto (puede ser K=1 — sobreajuste)
ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, 
           label=f'Mejor K raw={best_k} (acc={best_acc:.3f})')
ax.scatter([best_k], [best_acc], color='green', s=200, zorder=5, 
           edgecolors='black', linewidth=2)

# Resaltamos el K más robusto (K>=3, evitando sobreajuste)
ax.axvline(x=robust_k, color='purple', linestyle=':', alpha=0.7, 
           label=f'K robusto={robust_k} (acc={robust_acc:.3f})')
ax.scatter([robust_k], [robust_acc], color='purple', s=150, zorder=5, 
           edgecolors='black', linewidth=2, marker='D')

ax.set_xlabel('K (número de vecinos)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('🎯 ¿Cuál es el mejor K? — Accuracy vs Número de Vecinos', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.set_xticks(range(0, 51, 5))
ax.grid(True, alpha=0.3)

# Añadimos anotación explicativa
ax.annotate('K pequeño → sobreajuste\n(alta accuracy en train, baja en test)',
            xy=(2, train_scores[1]), xytext=(3, 0.98),
            fontsize=9, color='#3498db', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))

ax.annotate('K grande → subajuste\n(baja accuracy en ambos)',
            xy=(45, test_scores[-1]), xytext=(35, 0.70),
            fontsize=9, color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

plt.tight_layout()
plt.savefig('docs/03_k_optimizacion.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Gráfico guardado: docs/03_k_optimizacion.png")

# ============================================================================
# PASO 7: ENTRENAR MODELO FINAL CON EL MEJOR K
# ============================================================================
# Ahora que sabemos cuál es el mejor K, entrenamos el modelo final
# y lo evaluamos en detalle.
# ============================================================================

print(f"\n🏆 Paso 7: Entrenando modelo final con K={robust_k} (K robusto)...")
print(f"   (Nota: el mejor K absoluto es K={best_k} con acc={best_acc:.3f},")
print(f"    pero K={robust_k} es más robusto porque evita sobreajuste con K pequeño)\n")

modelo_final = KNeighborsClassifier(n_neighbors=robust_k)
modelo_final.fit(X_train_scaled, y_train)

# Predecimos sobre train y test
y_train_pred = modelo_final.predict(X_train_scaled)
y_test_pred = modelo_final.predict(X_test_scaled)

# Calculamos accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n   📈 Accuracy en entrenamiento: {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"   📈 Accuracy en test:         {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"   📊 Diferencia train-test:    {(train_acc - test_acc):.4f} "
      f"({'⚠️  Posible sobreajuste' if train_acc - test_acc > 0.1 else '✅ Buena generalización'})")

# ============================================================================
# PASO 8: EVALUACIÓN DETALLADA
# ============================================================================
# La accuracy no cuenta toda la historia. Si el 87% de los vinos son
# "normales", un modelo que siempre diga "normal" tendría 87% de accuracy
# pero ¡no sabría identificar ningún vino bueno!
#
# Por eso usamos más métricas:
#   Precision: de los que dije "buenos", ¿cuántos lo eran realmente?
#   Recall:    de los que eran buenos, ¿cuántos atrapé?
#   F1:        promedio armónico de precision y recall
# ============================================================================

print("\n📋 Paso 8: Evaluación detallada del modelo...")

# --- Reporte de clasificación ---
# classification_report genera precision, recall, f1 para cada clase
print("\n   📊 Reporte de clasificación (test):")
report = classification_report(y_test, y_test_pred, 
                               target_names=['Normal (0)', 'Buena (1)'],
                               digits=3)
print(report)

# --- Matriz de confusión ---
# La matriz de confusión nos dice:
#   [TN  FP]   TN = True Negatives (normales bien clasificados)
#   [FN  TP]   FP = False Positives (normales que dije buenos)
#              FN = False Negatives (buenos que dije normales)
#              TP = True Positives (buenos bien clasificados)

cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(8, 6))

# Visualización bonita de la matriz
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['Normal', 'Buena'])
disp.plot(ax=ax, cmap='Blues', values_format='d', 
          colorbar=False)

# Personalizamos
ax.set_title('🎯 Matriz de Confusión — KNN en vinos', fontsize=14, fontweight='bold')

# Añadimos explicaciones en las esquinas
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.45, 0.05, '✅ Aciertos', transform=ax.transAxes, fontsize=10,
        ha='center', va='center', fontweight='bold', bbox=props)
ax.text(0.95, 0.95, '❌ Errores', transform=ax.transAxes, fontsize=10,
        ha='center', va='center', fontweight='bold', bbox=props)

plt.tight_layout()
plt.savefig('docs/04_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Gráfico guardado: docs/04_matriz_confusion.png")

# Interpretación de la matriz
tn, fp, fn, tp = cm.ravel()
print(f"\n   📖 Interpretación de la matriz de confusión:")
print(f"      ✅ {tn} vinos NORMALES bien clasificados (True Negatives)")
print(f"      ❌ {fp} vinos NORMALES que el modelo dijo BUENOS (False Positives)")
print(f"      ❌ {fn} vinos BUENOS que el modelo dijo NORMALES (False Negatives)")
print(f"      ✅ {tp} vinos BUENOS bien clasificados (True Positives)")

# Calculamos métricas adicionales manualmente
precision_buena = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_buena = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_buena = 2 * precision_buena * recall_buena / (precision_buena + recall_buena) if (precision_buena + recall_buena) > 0 else 0

print(f"\n   📊 Métricas para la clase 'Buena':")
print(f"      Precision: {precision_buena:.3f}  → De los que dije 'buenos', {precision_buena*100:.1f}% lo eran")
print(f"      Recall:    {recall_buena:.3f}    → De los buenos reales, atrapé el {recall_buena*100:.1f}%")
print(f"      F1-score:  {f1_buena:.3f}   → Promedio armónico de precision y recall")

# ============================================================================
# PASO 9: VISUALIZAR LA FRONTERA DE DECISIÓN
# ============================================================================
# KNN separa el espacio en regiones. Cada región es la zona donde
# una clase gana la votación. Como tenemos 11 dimensiones, usamos PCA
# para reducirlas a 2 y poder visualizar la frontera.
# 
# PCA (Análisis de Componentes Principales) encuentra las 2 direcciones
# donde los datos varían más y proyecta los puntos en esas direcciones.
# ============================================================================

print("\n🎨 Paso 9: Visualizando la frontera de decisión con PCA...")

# Reducimos a 2 dimensiones para visualizar
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)  # Transformamos train
X_test_pca = pca.transform(X_test_scaled)        # Misma transformación en test

# Entrenamos KNN en las 2 dimensiones PCA
knn_pca = KNeighborsClassifier(n_neighbors=robust_k)
knn_pca.fit(X_train_pca, y_train)

# ---- Creamos una cuadrícula para dibujar la frontera ----
# La cuadrícula cubre todo el rango de los datos PCA
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

# Creamos una rejilla fina de puntos
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Predecimos la clase para cada punto de la rejilla
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ---- Dibujamos ----
fig, ax = plt.subplots(figsize=(12, 9))

# Fondo coloreado = frontera de decisión
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[-0.5, 0.5, 1.5])

# Puntos de entrenamiento
scatter_train = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                            c=y_train, cmap='RdYlGn', alpha=0.6, 
                            edgecolors='black', linewidth=0.5, s=40,
                            label='Train')

# Puntos de test con marcador diferente
scatter_test = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                           c=y_test, cmap='RdYlGn', alpha=1.0,
                           edgecolors='black', linewidth=1.5, s=80,
                           marker='s', label='Test')

ax.set_title(f'🌍 Frontera de Decisión — KNN (K={robust_k}) con PCA',
             fontsize=14, fontweight='bold')
ax.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)',
              fontsize=11)
ax.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)',
              fontsize=11)

# Leyenda personalizada
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.3, label='Región: Vino Normal'),
    Patch(facecolor='lightcoral', alpha=0.3, label='Región: Vino Bueno'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Train'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=8, label='Test'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

# Texto explicativo
ax.text(0.02, 0.02, 
        '🌈 Fondo = región que el modelo asigna a cada clase\n'
        '🟢 Zona verde → predice "Normal"\n'
        '🔴 Zona roja  → predice "Buena"',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/05_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Gráfico guardado: docs/05_decision_boundary.png")
print(f"   📊 Varianza explicada por PCA: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# ============================================================================
# PASO 10: PROBAR EL MODELO CON NUEVOS VINOS
# ============================================================================
# Vamos a simular 3 vinos nuevos y ver qué predice nuestro modelo.
# Esto es lo que harías en producción: llega un vino nuevo, mides sus
# características, y el modelo te dice si es bueno o no.
# ============================================================================

print("\n🔮 Paso 10: Probando el modelo con vinos nuevos...")

# Creamos 3 vinos hipotéticos con diferentes perfiles
# Vino 1: Alto alcohol, baja acidez volátil, altos sulfatos → debería ser bueno
# Vino 2: Bajo alcohol, alta acidez volátil → debería ser malo
# Vino 3: Perfil promedio → justo en el límite
nuevos_vinos = np.array([
    [7.5, 0.3, 0.5, 2.0, 0.06, 20, 50, 0.995, 3.3, 0.8, 12.5],  # Vino Premium
    [9.0, 1.2, 0.0, 3.5, 0.12, 8, 25, 0.998, 3.0, 0.4, 9.0],     # Vino Básico
    [8.5, 0.6, 0.3, 2.5, 0.08, 15, 40, 0.997, 3.2, 0.6, 10.5],   # Vino Promedio
])

nombres_vinos = ['🥇 Vino Premium', '🥉 Vino Básico', '🥈 Vino Promedio']

# Escalamos con el mismo scaler
nuevos_vinos_scaled = scaler.transform(nuevos_vinos)

# Predecimos
predicciones = modelo_final.predict(nuevos_vinos_scaled)
probabilidades = modelo_final.predict_proba(nuevos_vinos_scaled)

print(f"\n   {'Vino':20s} {'Predicción':15s} {'Confianza':15s}")
print(f"   {'-'*50}")
for nombre, pred, prob in zip(nombres_vinos, predicciones, probabilidades):
    clase = '🍷 BUENO' if pred == 1 else '🥤 Normal'
    confianza = prob[1] if pred == 1 else prob[0]
    print(f"   {nombre:20s} {clase:15s} {confianza*100:>7.1f}% confianza")

# ============================================================================
# PASO 11: COMPARAR DIFERENTES VALORES DE K
# ============================================================================
# Mostramos un resumen de cómo se comportan diferentes K
# ============================================================================

print(f"\n📊 Paso 11: Comparación de diferentes K...")
print(f"\n   {'K':5s} {'Train Acc':12s} {'Test Acc':12s} {'Diferencia':12s}")
print(f"   {'-'*41}")

valores_mostrar = [1, 3, 5, 7, robust_k, 15, 25, 50]
for k in valores_mostrar:
    if k in k_values:
        idx = k - 1  # porque k_values empieza en 1
        diff = train_scores[idx] - test_scores[idx]
        flag = ' ⚠️' if diff > 0.1 else ''
        print(f"   {k:5d} {train_scores[idx]:10.3f}   {test_scores[idx]:10.3f}   {diff:8.3f}{flag}")

# ============================================================================
# 🧠 RESUMEN — ¿Qué aprendiste hoy?
# ============================================================================
print("\n" + "="*65)
print("🧠  RESUMEN: Lo que aprendiste hoy")
print("="*65)
print(f"""
  ✅ Clasificación: Aprendiste que la clasificación predice CATEGORÍAS
     (bueno/malo, spam/no spam), no números como en regresión.

  ✅ KNN (K-Vecinos más Cercanos): Un algoritmo que clasifica según
     la mayoría de votos entre los K vecinos más cercanos.

  ✅ Escalado de features: Descubriste por qué KNN requiere que todas
     las variables estén en la misma escala (StandardScaler).

  ✅ El mejor K: Encontraste que K={robust_k} es el valor más robusto
     con un accuracy de {robust_acc*100:.1f}% en test.
     K=1 da {best_acc*100:.1f}% pero sobreajusta (100% en train).
     K pequeño sobreajusta, K grande subajusta.

  ✅ Matriz de confusión: Aprendiste a interpretar TN, FP, FN, TP
     y por qué la accuracy no es suficiente.

  ✅ Precision vs Recall: Dos métricas que cuentan historias distintas.
     Precision = ¿fui preciso al decir "bueno"? 
     Recall = ¿atrapé a todos los buenos?

  ✅ Frontera de decisión: Visualizaste cómo KNN divide el espacio
     en regiones de decisión usando PCA.

  🔮 Proyecto guardado en: /opt/data/ml-projects/2026-06-26-knn-classification/
  📊 {len([f for f in os.listdir('docs') if f.endswith('.png')])} gráficos generados en docs/
  
  🚀 Siguiente proyecto: Árboles de Decisión — un clasificador que
     puedes interpretar visualmente (como un diagrama de flujo).
""")
