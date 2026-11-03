# Protocolo de modelado y evaluación

## Pipeline canónico

La ruta oficial del proyecto es `scripts/train_ml.py`. Los notebooks sirven como material explicativo y exploratorio; no deben sobrescribir `output/metrics/model_compare.json` ni sustituir el script canónico.

## Variables

El ETL conserva `income_yearly` y `overtime_flag` para facilitar análisis descriptivos y persistencia. Ambas son transformaciones deterministas de variables ya presentes (`MonthlyIncome` y `OverTime`) y se excluyen del entrenamiento para evitar duplicidad de información.

La columna `Attrition` se transforma en `attrition_label` y después se elimina antes del modelado. Las columnas `survey_*` se conservan para demostrar la integración de fuentes, pero también se excluyen del entrenamiento.

## Evaluación

Se utiliza validación cruzada estratificada de 5 particiones. Para cada partición externa:

1. El conjunto de entrenamiento externo se usa para seleccionar el umbral mediante una CV interna de 3 particiones.
2. El modelo se ajusta de nuevo sobre todo el entrenamiento externo.
3. Se predice el pliegue externo, que permanece fuera del ajuste y de la selección del umbral.

Las métricas ROC-AUC, PR-AUC y F1 evaluado se calculan sobre las predicciones externas agrupadas. Así se evita seleccionar el umbral sobre las mismas predicciones que después se utilizan para medir el F1.

El `threshold_reference` es la mediana de los cinco umbrales seleccionados mediante CV interna en los pliegues externos. Se utiliza como corte común para contar señales OOF y para el modelo final persistido. El F1 evaluado, en cambio, promedia los cinco F1 externos con el umbral específico de cada fold; el corte común no se usa para calcular ese F1.

## Interpretación

Los coeficientes de Regresión Logística representan asociaciones condicionadas por el conjunto de variables y su codificación. No deben describirse como efectos causales.

## Separación entre evaluación y predicciones persistidas

El panel HTML presenta las métricas calculadas sobre predicciones OOF, las tasas observadas, los recuentos de señales OOF y los factores asociados en apartados separados. Las predicciones del pipeline final ajustado con todos los registros se guardan en PostgreSQL y constituyen una salida distinta; no se usan para calcular las métricas OOF ni las señales de evaluación.

El umbral de referencia permite construir los recuentos OOF que acompañan la evaluación y configurar las predicciones del modelo final. Los informes y gráficos identifican por separado cada tipo de resultado.

## Encuesta sintética

Todas las columnas `survey_*` se excluyen del entrenamiento. Su finalidad es demostrar integración de fuentes y disponibilidad de campos para el flujo de datos.
