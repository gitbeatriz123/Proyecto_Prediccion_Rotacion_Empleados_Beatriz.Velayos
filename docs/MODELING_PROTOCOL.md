# Protocolo de modelado y evaluación

## Pipeline canónico

La ruta oficial del proyecto es 'scripts/train_ml.py'. Los notebooks sirven como material explicativo y exploratorio; no deben sobrescribir 'output/metrics/model_compare.json' ni sustituir al script canónico.

## Variables

El ETL conserva 'income_yearly' y 'overtime_flag' para facilitar análisis y persistencia en BI. Sin embargo, ambas son transformaciones deterministas de variables ya presentes ('MonthlyIncome' y 'OverTime') y se excluyen del entrenamiento para evitar duplicidad de información.

La columna 'Attrition' se transforma en 'attrition_label' y después se elimina antes del modelado.

## Evaluación

Se utiliza validación cruzada estratificada de 5 particiones. Para cada partición externa:

1. El conjunto de entrenamiento externo se usa para seleccionar el umbral mediante una CV interna de 3 particiones.
2. El modelo se ajusta de nuevo sobre todo el entrenamiento externo.
3. Se predice el pliegue externo, que permanece fuera del ajuste y de la selección del umbral.

Las métricas ROC-AUC, PR-AUC y F1 evaluado se calculan sobre las predicciones externas agrupadas. Así se evita seleccionar el umbral sobre las mismas predicciones que después se utilizan para medir el F1.

El 'threshold_reference' que aparece en los artefactos es la mediana de los umbrales seleccionados en los cinco pliegues externos. Se utiliza únicamente como referencia operativa para las visualizaciones; no es una estimación de rendimiento independiente.

## Interpretación

Los coeficientes de Regresión Logística representan asociaciones condicionadas por el conjunto de variables y su codificación. No deben describirse como efectos causales.
