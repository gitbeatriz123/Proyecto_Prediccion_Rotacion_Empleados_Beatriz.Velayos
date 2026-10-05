# Resultados finales verificados

Esta es la ejecución canónica del proyecto tras aplicar el protocolo de validación cruzada anidada.

## Protocolo

- 5 folds externos estratificados.
- 3 folds internos para seleccionar el umbral dentro de cada entrenamiento externo.
- El fold externo queda fuera del ajuste y de la selección del umbral.
- F1 evaluado = media de los F1 obtenidos en los cinco folds externos.
- Las columnas `survey_*` son sintéticas y quedan excluidas del entrenamiento.
- `income_yearly` y `overtime_flag` se excluyen por ser transformaciones deterministas de variables ya existentes.

## Métricas

| Modelo | ROC-AUC | PR-AUC | F1 evaluado | Umbral de referencia | Exactitud a 0,5 | F1 a 0,5 |
|---|---:|---:|---:|---:|---:|---:|
| Regresión Logística | 0,823 | 0,575 | 0,542 | 0,706 | 0,748 | 0,472 |
| Random Forest | 0,803 | 0,540 | 0,523 | 0,257 | 0,855 | 0,242 |
| MLP | 0,814 | 0,559 | 0,522 | 0,316 | 0,868 | 0,458 |

La Regresión Logística es el modelo seleccionado para la lectura principal porque lidera ROC-AUC, PR-AUC y F1 evaluado.

## Datos descriptivos

- 1.470 registros.
- 237 casos positivos.
- Prevalencia de Attrition: 16,1 %.
- Ventas: 20,6 % (92/446).
- Recursos Humanos: 19,0 % (12/63).
- Investigación y Desarrollo: 13,8 % (133/961).
- Con horas extra: 30,5 % (127/416).
- Sin horas extra: 10,4 % (110/1.054).

## Señales OOF

Con el umbral de referencia 0,706 aplicado a las predicciones OOF de Regresión Logística:

- 251 señales en total.
- Ventas: 110.
- Investigación y Desarrollo: 131.
- Recursos Humanos: 10.

Estas cifras son de evaluación y no deben confundirse con las predicciones operativas generadas por el pipeline final ajustado con todos los registros.

## Interpretación responsable

Los coeficientes de la Regresión Logística son asociaciones condicionadas por las variables y su codificación. No son efectos causales. Las señales del modelo deben utilizarse como apoyo para análisis agregado y priorización de preguntas, nunca como decisiones automáticas sobre personas.

## Verificación reproducible

La ejecución de GitHub Actions ha completado correctamente:

1. generación de encuesta sintética;
2. control de calidad;
3. ETL;
4. entrenamiento canónico;
5. tests;
6. generación de previews;
7. publicación de artefactos de reproducibilidad.

El PBIX final debe refrescarse en Power BI Desktop contra PostgreSQL para asegurar que su snapshot de datos coincide con esta ejecución.
