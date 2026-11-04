# Panel de rotación de empleados

Esta página muestra las gráficas directamente en GitHub. Haz clic en una imagen para verla a tamaño completo.

## Comparación de modelos

Compara ROC-AUC, PR-AUC y F1 evaluado para Regresión Logística, Random Forest y MLP. Regresión Logística obtiene los valores más altos en las tres métricas.

[![Comparación de métricas de los modelos](01_Resumen_KPIs.png)](01_Resumen_KPIs.png)

## Distribución de puntuaciones

Muestra cómo se distribuyen las puntuaciones OOF de Regresión Logística entre los registros.

[![Distribución de puntuaciones OOF](02_Distribucion_Puntuaciones.png)](02_Distribucion_Puntuaciones.png)

## Rotación por departamento

Presenta la tasa observada, el número de salidas y las señales OOF de cada departamento para facilitar la comparación de sus tamaños y resultados.

[![Tasa observada de rotación por departamento](03_Departamentos_Tasa_Rotacion.png)](03_Departamentos_Tasa_Rotacion.png)

## Rotación y horas extra

Compara las tasas observadas y sus denominadores para los grupos con y sin horas extra.

[![Tasa de rotación según horas extra](04_Tasa_Rotacion_Horas_Extra.png)](04_Tasa_Rotacion_Horas_Extra.png)

## Factores asociados al modelo

Muestra los coeficientes de Regresión Logística con mayor magnitud. Sus valores describen asociaciones del modelo.

[![Factores asociados según Regresión Logística](05_Factores_Asociados_Modelo.png)](05_Factores_Asociados_Modelo.png)

[Consultar resultados y recomendaciones](../RESULTADOS_FINALES.md)

La versión HTML con selector de modelos se conserva como archivo descargable en esta carpeta: `Panel_Rotacion_Empleados_interactivo.html`.