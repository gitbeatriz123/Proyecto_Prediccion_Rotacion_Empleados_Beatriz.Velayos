# Resultados finales verificados

Esta es la ejecución canónica del proyecto tras aplicar el protocolo de validación cruzada anidada.

## Protocolo

- 5 folds externos estratificados.
- 3 folds internos para seleccionar el umbral dentro de cada entrenamiento externo.
- El fold externo queda fuera del ajuste y de la selección del umbral.
- F1 evaluado = media de los F1 obtenidos en los cinco folds externos; cada fold utiliza el umbral seleccionado en sus tres folds internos.
- Las columnas `survey_*` son sintéticas y quedan excluidas del entrenamiento.
- `income_yearly` y `overtime_flag` se excluyen por ser transformaciones deterministas de variables ya existentes.

## Métricas

| Modelo | ROC-AUC | PR-AUC | F1 evaluado | Umbral de referencia (mediana) | Exactitud a 0,5 | F1 a 0,5 |
|---|---:|---:|---:|---:|---:|---:|
| Regresión Logística | 0,823 | 0,575 | 0,542 | 0,706 | 0,748 | 0,472 |
| Random Forest | 0,803 | 0,540 | 0,523 | 0,257 | 0,855 | 0,242 |
| MLP | 0,814 | 0,559 | 0,522 | 0,316 | 0,868 | 0,458 |

La Regresión Logística lidera las tres métricas de evaluación: ROC-AUC, PR-AUC y F1 evaluado. El F1 evaluado promedia los cinco folds externos con sus umbrales internos; la columna de umbral muestra la mediana usada como referencia común.

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

Con el umbral de referencia de Regresión Logística (mediana de los cinco umbrales externos, 0,706) aplicado a las 1.470 puntuaciones OOF agrupadas:

- 251 señales en total.
- Ventas: 110.
- Investigación y Desarrollo: 131.
- Recursos Humanos: 10.

Estas señales son recuentos sobre puntuaciones OOF. El panel HTML y el libro de datos presentan métricas, tasas observadas, señales y factores en apartados separados. El libro permite crear un PBIX actual en Power BI Desktop sin conectar Docker.

## Lectura de negocio

- **Ventas:** comenzar el diagnóstico por su tasa observada más alta (20,6 %, 92/446) y sus 110 señales OOF. Revisar viajes frecuentes, objetivos y carga comercial con la dirección del área.
- **Investigación y Desarrollo:** desglosar por equipo y puesto; concentra 133 salidas observadas y 131 señales OOF, aunque su tasa sea 13,8 %.
- **Organización del trabajo:** revisar planificación, turnos y reparto de carga donde hay horas extra; comparar la tasa observada de 30,5 % con el 10,4 % del grupo sin horas extra.
- **Recursos Humanos:** acordar una revisión proporcional al tamaño del área (12 salidas de 63 personas y 10 señales OOF).
- **Seguimiento:** registrar una línea de partida, asignar responsables y revisar cada mes las tasas, salidas, señales y denominadores para valorar el avance de los pilotos.

Los coeficientes describen asociaciones del modelo y ayudan a formular preguntas para el diagnóstico. Las intervenciones se evalúan con la evolución de los indicadores observados.

## Verificación reproducible

La ejecución canónica de GitHub Actions [37276469074](https://github.com/gitbeatriz123/Proyecto_Prediccion_Rotacion_Empleados_Beatriz.Velayos/actions/runs/37276469074) completó correctamente:

1. generación de encuesta sintética;
2. control de calidad;
3. ETL;
4. entrenamiento canónico;
5. tests;
6. generación de previews;
7. publicación de artefactos de reproducibilidad.

El [panel interactivo HTML](powerbi/Panel_Rotacion_Empleados_interactivo.html) refleja estos resultados y puede abrirse sin Docker. El [libro Excel de datos para Power BI](../bi/Panel_PowerBI_datos_canonicos.xlsx) contiene las tablas agregadas para crear el PBIX en Power BI Desktop.
