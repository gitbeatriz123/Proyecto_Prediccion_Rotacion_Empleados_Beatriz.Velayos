# Predicción de rotación de empleados

Proyecto educativo de Big Data y analítica de Recursos Humanos. Integra preparación de datos con PySpark, modelado con scikit-learn y visualización en Power BI. Los resultados se interpretan para una empresa hipotética y sus áreas de negocio.

## Alcance

- Control de calidad e integración de dos ficheros tabulares mediante un proceso ETL con PySpark.
- Análisis exploratorio y comparación de Regresión Logística, Random Forest y MLP.
- Métricas, puntuaciones OOF, tasas descriptivas y factores asociados a la rotación.
- Recomendaciones por departamento y una propuesta de seguimiento en Power BI.

**Tecnologías:** Python, PySpark, scikit-learn, Jupyter, Docker Compose, PostgreSQL y Power BI.

## Entregables

- [Panel de Power BI](bi/Panel_Rotacion_Empleados.pbix)
- [Presentación del proyecto](docs/Presentacion_Proyecto_Rotacion.pptx)
- [Memoria del proyecto (PDF)](docs/Memoria_Proyecto_Rotacion.pdf) · [versión HTML](docs/Memoria_Proyecto_Rotacion.html)
- Notebooks de análisis, modelado e informe final en `notebooks/`
- [Documentación de arquitectura](docs/arquitectura.md)

## Panel de indicadores

![Comparación de métricas de los modelos](docs/powerbi/01_Resumen_KPIs.png)

Los gráficos de `docs/powerbi/` resumen indicadores y resultados. El PBIX contiene el panel con vistas por departamento, puesto y horas extra. La presentación incluye gráficos editables y una descripción breve en castellano de cada visualización.

## Datos y resultados

El CSV principal reúne 1.470 registros y 35 variables sobre características laborales y rotación. La memoria incluye una referencia de contexto de IBM sobre clasificación de rotación. El fichero `encuesta_clima.csv` se construye mediante `scripts/generate_synthetic_survey.py`: asigna cinco escalas aleatorias de 1 a 5 con semilla 42, sin consultar Attrition, y permite reproducir la unión y el proceso ETL.

La evaluación compara tres clasificadores mediante validación cruzada estratificada de cinco particiones y puntuaciones *out-of-fold* (OOF). Regresión Logística alcanza ROC-AUC 0,819 y PR-AUC 0,552; Random Forest obtiene el mayor F1 comparativo (0,544). El umbral de Regresión Logística usado para ordenar las señales agregadas es 0,732: en las predicciones OOF aparecen 104 señales en Ventas, 115 en Investigación y Desarrollo y 9 en Recursos Humanos, 228 en total. Estos recuentos se leen junto con las tasas observadas y el tamaño de cada área.

Los archivos de `output/` contienen métricas, modelos, gráficos y exportaciones asociadas a los datos incluidos. Las instrucciones siguientes permiten regenerar la encuesta, reconstruir el ETL y volver a calcular los resultados. Las figuras de docs/powerbi/ se regeneran desde estas salidas con scripts/render_dashboard_previews.py.

## Ejecución local

Requisitos: Docker Engine y Docker Compose v2 (`docker compose`), además de espacio para descargar las imágenes y dependencias.

1. Clona el repositorio y entra en la carpeta.
2. Construye y arranca los servicios:

   ```bash
   docker compose up -d --build
   ```

3. Obtén el token temporal generado por Jupyter:

   ```bash
   docker compose logs jupyter
   ```

4. Abre `http://localhost:8889/lab` e introduce el token mostrado en los registros. El token no se guarda en el repositorio.
5. Regenera la encuesta sintética cuando quieras reconstruirla con la semilla documentada:

   ```bash
   docker compose run --rm jupyter python /scripts/generate_synthetic_survey.py \
     --input /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --output /data/raw/encuesta_clima.csv --seed 42
   ```

6. Revisa los datos y ejecuta el ETL:

   ```bash
   docker compose run --rm jupyter python /scripts/check_data_quality.py \
     --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --input2 /data/raw/encuesta_clima.csv \
     --key EmployeeNumber --max-null-frac 0.25

   docker compose run --rm jupyter python /scripts/etl_attrition.py \
     --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --input2 /data/raw/encuesta_clima.csv \
     --outdir /data/processed/employee_attrition.parquet
   ```

7. Entrena los modelos y, si deseas recorrer el análisis, abre los notebooks en este orden: `01_EDA_Attrition`, `02_Modelado_Baseline`, `03_Modelado_DL`, `05_Dashboard_KPIs` y `99_Informe_Final`.

   ```bash
   docker compose run --rm jupyter python /scripts/train_ml.py \
     --input /data/processed/employee_attrition.parquet --model all

   docker compose run --rm jupyter python /scripts/render_dashboard_previews.py
   ```

Para detener los servicios: `docker compose down`. El ETL usa Spark en modo local por defecto. Docker Compose reúne Jupyter, Spark Master y PostgreSQL; el entorno también puede ampliarse con workers para procesamiento distribuido.

## Recomendaciones para la empresa hipotética

Los resultados sitúan Ventas en primer lugar por su tasa observada (20,6 %, 92 de 446 registros) y 104 señales OOF. Investigación y Desarrollo aporta el mayor volumen de salidas (133) y reúne 115 señales, con una tasa del 13,8 %. Recursos Humanos presenta 19,0 % (12 de 63) y 9 señales. La diferencia por horas extra —30,5 % frente a 10,4 %— señala un segundo eje de actuación transversal.

- **Ventas:** revisar con su dirección los viajes frecuentes, los objetivos y la distribución de carga en puestos comerciales; acordar un piloto y seguir la tasa y el número de casos.
- **Investigación y Desarrollo:** desglosar las 133 salidas por equipo y puesto, y concentrar las medidas de acompañamiento donde el volumen sea mayor.
- **Organización del trabajo:** revisar turnos, cobertura y reparto de tareas en los equipos con horas extra; comparar las tasas antes y después de los ajustes.
- **Recursos Humanos:** planificar la revisión teniendo presente el tamaño del área y el detalle de puestos, para interpretar adecuadamente su tasa del 19,0 %.

Para el primer ciclo, combinar un piloto en Ventas y en los equipos con más horas extra con un análisis de puestos y equipos de I+D. En el panel, seguir mensualmente tasa, número de salidas, señales y denominadores por área y puesto. Regresión Logística ofrece la mejor ordenación global (ROC-AUC 0,819; PR-AUC 0,552); Random Forest alcanza el F1 comparativo más alto (0,544) y completa la comparación según el criterio de negocio.

## Estructura

```text
bi/                Panel de Power BI
data/raw/           Datos de entrada
docs/               Memoria, presentación, arquitectura y gráficos
notebooks/          Análisis, modelado e informe
output/             Métricas, modelos, gráficos y exportaciones
scripts/             ETL, controles y entrenamiento
docker-compose.yml  Servicios locales
```

## Licencia y reutilización

El repositorio no declara una licencia general de reutilización. Antes de redistribuir por separado el conjunto HR incluido, consulta las condiciones aplicables a esa copia. La encuesta sintética puede regenerarse con el script y la semilla documentada.
