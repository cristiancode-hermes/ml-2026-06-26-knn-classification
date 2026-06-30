# Learnings — 2026-06-26

## Qué se hizo
Proyecto de clasificación con K-Nearest Neighbors (KNN) usando el dataset
winequality-red (vinos tintos). Se predijo si un vino es "bueno" (quality >= 7)
basado en 11 características químicas.

## Conceptos nuevos
- [x] Clasificación: predecir categorías vs números (regresión)
- [x] KNN: algoritmo de vecinos más cercanos, basado en distancia
- [x] StandardScaler: escalado de features (crítico para KNN)
- [x] train_test_split con stratify (preservar proporción de clases)
- [x] Accuracy, Matriz de Confusión (TN, FP, FN, TP)
- [x] Precision, Recall, F1-score
- [x] Búsqueda del mejor K (hiperparámetro)
- [x] Frontera de decisión con PCA (visualización 2D)
- [x] Sobreajuste con K pequeño vs subajuste con K grande
- [x] Desbalanceo de clases (~87% normal vs ~13% bueno)

## Dificultades encontradas
- Ninguna durante la creación del proyecto
- El dataset usa ; como separador (no ,) — importante recordarlo
- El desbalanceo de clases afecta la precisión de la clase minoritaria
- PCA pierde ~40% de varianza al reducir a 2D (11 dimensiones originales)

## Notas para el próximo proyecto
- Siguiente: Árboles de Decisión (otro clasificador, más interpretable)
- Después: Random Forest (ensamble de árboles)
- Feature engineering podría mejorar KNN (seleccionar mejores features)
- Probar weighted KNN (vecinos más cercanos pesan más)
