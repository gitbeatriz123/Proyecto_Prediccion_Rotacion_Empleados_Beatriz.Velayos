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

El `threshold_reference` es la mediana de los cinco umbrales seleccionados mediante CV interna en los pliegues externos. Se utiliza como corte común para contar señales OOF y para el modelo final persistido. El F1 evaluado, en cambio, promedia los cinco F1 externos con el umbral específico de cada fold; el corte común no se usa para calcular ese F1.

## Interpretación

Los coeficientes de Regresión Logística representan asociaciones condicionadas por el conjunto de variables y su codificación. No deben describirse como efectos causales.


## Separación entre evaluación y operación

Las métricas ROC-AUC, PR-AUC y F1 evaluado proceden de las predicciones de los folds externos. Las vistas del PBIX se alimentan con probabilidades `proba` y etiquetas `pred` del pipeline final ajustado con todos los registros. La medida `risk_rate` resume la proporción de predicciones positivas del modelo. No equivale a la tasa observada de Attrition ni al recuento de señales OOF; el panel debe rotularla como riesgo predicho.

El umbral de referencia permite construir los recuentos OOF que acompañan la evaluación y configurar las predicciones del modelo final. Las dos salidas se distinguen en informes, gráficos y dashboard. La copia versionada del PBIX mantiene un snapshot previo; sigue la guía de refresco para actualizar las métricas y el selector de modelos.

## Encuesta sintética

Todas las columnas `survey_*` se excluyen del entrenamiento. Su única finalidad es demostrar integración de fuentes y disponibilidad de campos para el flujo de datos/BI.
