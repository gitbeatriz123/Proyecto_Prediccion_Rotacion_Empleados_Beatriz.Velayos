# Predicción de rotación de empleados

Proyecto educativo de Big Data y analítica de Recursos Humanos. Recorre la preparación de datos con PySpark, el análisis en Jupyter, la comparación de modelos con scikit-learn y la presentación de resultados en un prototipo de Power BI.

> **Uso formativo:** los datos y resultados son de demostración y exploratorios. No son una herramienta validada para evaluar, seleccionar, retener ni gestionar empleados, y no deben usarse con datos de una empresa ni para tomar decisiones sobre personas.

## Contenido

- Controles de calidad y proceso ETL en PySpark.
- Notebooks de análisis exploratorio y modelado.
- Comparación de Regresión Logística, Random Forest y MLP.
- Persistencia de artefactos y resultados en PostgreSQL y CSV.
- Memoria y presentación con resultados, explicación de gráficos y límites.
- Prototipo PBIX y gráficos estáticos de referencia.

**Tecnologías:** Python, PySpark, scikit-learn, Jupyter, Docker Compose, PostgreSQL y Power BI.

## Dashboard

![Resumen de KPIs del prototipo](docs/powerbi/01_Resumen_KPIs.png)

La imagen y los PNG de `docs/powerbi/` son gráficos estáticos preparados para mostrar resúmenes en GitHub. No son capturas de la aplicación Power BI ni prueban que se hayan comprobado filtros o interacciones. El archivo PBIX es un prototipo editable: para revisar sus páginas, filtros, relaciones y cifras hay que abrirlo en Power BI Desktop. La presentación incluida contiene gráficos editables con descripciones en castellano.

## Entregables

- [Prototipo de Power BI (PBIX)](bi/Panel_Rotacion_Empleados.pbix)
- [Presentación del proyecto (PPTX)](docs/Presentacion_Proyecto_Rotacion.pptx)
- [Memoria del proyecto (PDF)](docs/Memoria_Proyecto_Rotacion.pdf) · [versión HTML](docs/Memoria_Proyecto_Rotacion.html)
- Notebooks de análisis, modelado e informe final en `notebooks/`
- [Documentación de arquitectura](docs/arquitectura.md)

## Ejecución local

Requisitos: Docker Engine y Docker Compose v2 (`docker compose`), además de espacio para descargar las imágenes y dependencias.

1. Clona el repositorio y entra en la carpeta.
2. Construye y arranca los servicios:

   ```bash
   docker compose up -d --build
   ```

3. Obtén el token temporal que Jupyter genera al iniciar:

   ```bash
   docker compose logs jupyter
   ```

4. Abre `http://localhost:8889/lab` y pega el token. El token no se guarda en el repositorio.
5. Genera la encuesta sintética si quieres reconstruirla con la semilla publicada:

   ```bash
   docker compose run --rm jupyter python /scripts/generate_demo_survey.py \
     --input /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --output /data/raw/encuesta_clima.csv --seed 42
   ```

   Las cinco escalas de la encuesta se asignan al azar. No son respuestas de personas ni mediciones reales de clima.

6. Ejecuta controles de datos y ETL:

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

7. Entrena los tres modelos y abre los notebooks en este orden: `01_EDA_Attrition`, `02_Modelado_Baseline`, `03_Modelado_DL`, `05_Dashboard_KPIs` y `99_Informe_Final`.

   ```bash
   docker compose run --rm jupyter python /scripts/train_ml.py \
     --input /data/processed/employee_attrition.parquet --model all
   ```

   La comparación emplea predicciones *out-of-fold* de validación cruzada estratificada de cinco particiones. El F1 y el umbral se seleccionan sobre esas mismas predicciones; por ello, son resultados exploratorios y no una estimación independiente del rendimiento futuro.

Para detener los servicios: `docker compose down`. El ETL usa Spark en modo local por defecto; el servicio Spark Master no tiene un Worker configurado.

## Datos, resultados y límites

El CSV principal se parece al conjunto conocido como IBM HR Analytics Employee Attrition & Performance. Un tutorial de IBM presenta un ejercicio de clasificación de rotación, pero esa referencia no confirma la procedencia exacta ni la licencia de redistribución de la copia incluida aquí. [Consultar el tutorial de IBM](https://developer.ibm.com/caas-storage/skillscollection/dna/live/innovator-predict-employee-turnover-using-ibm-watson-studio/en/_attachments/Build-train-and-evaluate-Machine-Learning-models.pdf).

`encuesta_clima.csv` se genera con `scripts/generate_demo_survey.py`: cinco escalas aleatorias entre 1 y 5, con semilla fija 42 y sin consultar la etiqueta de rotación. No contiene respuestas reales ni una medición observada de clima.

Las métricas versionadas en `output/metrics/model_compare.json` se calcularon con el mismo esquema OOF para los tres modelos. Regresión Logística, Random Forest y MLP obtienen ROC-AUC de 0,819 / 0,810 / 0,768; PR-AUC de 0,552 / 0,532 / 0,484; y F1 optimizado de 0,533 / 0,544 / 0,470, respectivamente. El umbral y el F1 se seleccionan en las mismas predicciones OOF, lo que introduce optimismo. La memoria explica la lectura de estos resultados y las tasas descriptivas.

En esta revisión, las métricas se regeneraron usando un Parquet local construido con las transformaciones y nombres de columnas del ETL. Docker Compose y PySpark no estaban disponibles en ese entorno, por lo que no se ejecutó de extremo a extremo la cadena Docker/Spark.

No se hizo validación temporal, calibración independiente, análisis formal de equidad ni validación para decisiones laborales. El rendimiento en este conjunto de ejemplo no predice el de otra organización.

## Estructura

```text
bi/                 Prototipo Power BI
data/raw/           Datos de entrada del ejercicio
docs/               Memoria, presentación, arquitectura y gráficos
notebooks/          Análisis, modelado e informe
output/             Métricas, modelos y exportaciones de referencia
scripts/             ETL, comprobaciones y entrenamiento
docker-compose.yml  Servicios locales
```

Los artefactos de `output/` son resultados de referencia. Vuelve a generarlos si cambias los datos, el código o las dependencias.

## Licencia y reutilización

El repositorio no declara una licencia general de reutilización. La encuesta sintética puede regenerarse desde el script. Antes de redistribuir por separado el conjunto HR incluido, verifica las condiciones aplicables a esa copia.

## Uso responsable

El proyecto es formativo y sus datos no representan a ninguna plantilla real. No introduzcas datos personales, confidenciales o de empleados. Los patrones del conjunto no prueban causalidad. Cualquier estudio con datos laborales reales necesitaría evaluación independiente y revisión de privacidad, sesgo, equidad y contexto.
