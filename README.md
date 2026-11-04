# Predicción de rotación de empleados

Proyecto educativo de Big Data y analítica de Recursos Humanos. Integra preparación de datos con PySpark, comparación de modelos con scikit-learn y un panel interactivo para analizar una empresa hipotética y orientar prioridades de negocio.

## Qué incluye

- Control de calidad, integración de datos y ETL con PySpark.
- Análisis exploratorio y comparación de Regresión Logística, Random Forest y MLP.
- Evaluación con validación cruzada anidada y selección de umbral dentro de los datos de entrenamiento.
- Persistencia de características y predicciones en PostgreSQL.
- Panel interactivo en HTML, figuras de resultados, recomendaciones para dirección y documentación del flujo.

**Tecnologías:** Python, PySpark, scikit-learn, Jupyter, Docker Compose y PostgreSQL.

## Entregables

- [Panel interactivo en HTML](docs/panel/Panel_Rotacion_Empleados_interactivo.html): descarga el archivo y ábrelo con un navegador para usar el selector de modelo.
- [Presentación del proyecto](docs/Presentacion_Proyecto_Rotacion.pptx).
- [Resultados finales verificados](docs/RESULTADOS_FINALES.md).
- [Memoria del proyecto (PDF)](docs/Memoria_Proyecto_Rotacion.pdf) · [versión HTML](docs/Memoria_Proyecto_Rotacion.html).
- Notebooks de análisis y documentación en `notebooks/`.
- [Arquitectura](docs/arquitectura.md), [procedencia de datos](docs/DATA_SOURCE.md) y [protocolo de modelado](docs/MODELING_PROTOCOL.md).

## Datos

El dataset principal contiene 1.470 registros y 35 variables de rotación de empleados. La procedencia y las limitaciones están documentadas en `docs/DATA_SOURCE.md`.

`encuesta_clima.csv` se genera con semilla 42 y cinco escalas aleatorias de 1 a 5, sin consultar la etiqueta de rotación. Las variables `survey_*` se excluyen del entrenamiento.

## Pipeline reproducible

Desde la raíz del repositorio, con Docker Engine y Docker Compose v2:

```bash
docker compose up -d --build

docker compose run --rm jupyter python /scripts/generate_synthetic_survey.py \
  --input /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --output /data/raw/encuesta_clima.csv --seed 42

docker compose run --rm jupyter python /scripts/check_data_quality.py \
  --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --input2 /data/raw/encuesta_clima.csv \
  --key EmployeeNumber --max-null-frac 0.25

docker compose run --rm jupyter python /scripts/etl_attrition.py \
  --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --input2 /data/raw/encuesta_clima.csv \
  --outdir /data/processed/employee_attrition.parquet

docker compose run --rm jupyter python /scripts/train_ml.py \
  --input /data/processed/employee_attrition.parquet --model all

docker compose run --rm jupyter python /scripts/load_features_to_pg.py \
  --input /data/processed/employee_attrition.parquet

for MODEL in LogReg RF MLP; do
  case "$MODEL" in
    LogReg) ART="/output/models/logreg_pipeline.pkl" ;;
    RF) ART="/output/models/rf_pipeline.pkl" ;;
    MLP) ART="/output/models/mlp_pipeline.pkl" ;;
  esac
  docker compose run --rm jupyter python /scripts/persist_results.py \
    --input /data/processed/employee_attrition.parquet \
    --model "$ART" --model_name "$MODEL" \
    --metrics_json /output/metrics/model_compare.json
done

docker compose run --rm jupyter python /scripts/render_dashboard_previews.py
```

La evaluación usa 5 folds externos y 3 folds internos para seleccionar el umbral. Las variables `survey_*` se conservan para recorrer la integración de los dos ficheros, pero no se usan para entrenar. Las métricas canónicas se guardan en `output/metrics/`; las predicciones OOF y los coeficientes asociados, en `output/dashboard/`; y las figuras del panel se regeneran en `docs/panel/`.

## PostgreSQL

El esquema reproducible está en `postgres/init/001_schema.sql`. Las credenciales y los puertos se pueden configurar mediante `.env`, usando `.env.example` como plantilla.

Para persistir resultados:

```bash
docker compose run --rm jupyter python /scripts/load_features_to_pg.py \
  --input /data/processed/employee_attrition.parquet

docker compose run --rm jupyter python /scripts/persist_results.py \
  --input /data/processed/employee_attrition.parquet \
  --model /output/models/logreg_pipeline.pkl \
  --model_name LogReg \
  --metrics_json /output/metrics/model_compare.json
```

## Notebooks

Para regenerar sus HTML desde cero: `bash scripts/export_notebooks.sh`.

Los notebooks documentan el análisis. `02_Modelado_Baseline` y `03_Modelado_DL` muestran análisis exploratorios independientes; sus resultados no sustituyen la evaluación canónica de `scripts/train_ml.py`.

## Panel interactivo y visualizaciones

![Comparación de métricas de los modelos](docs/panel/01_Resumen_KPIs.png)

El [panel HTML interactivo](docs/panel/Panel_Rotacion_Empleados_interactivo.html) ofrece un selector de modelo y presenta las métricas comparativas, tasas observadas, señales OOF, factores asociados y recomendaciones. Para usar la interacción, descarga el archivo HTML y ábrelo en un navegador. Las figuras de `docs/panel/` se regeneran a partir de los artefactos canónicos y distinguen las tasas observadas de las señales OOF.

## Reproducibilidad y calidad

- Dependencias Python fijadas en `requirements-ml.txt`.
- Versiones de Docker fijadas a tags concretos.
- Healthcheck de PostgreSQL antes de arrancar Jupyter.
- Tests unitarios en `tests/`.
- CI en `.github/workflows/ci.yml`.
- Salidas generables en `output/metrics`, `output/models`, `output/plots` y `output/dashboard`.

## Interpretación

Las tasas observadas, las métricas de evaluación, las señales OOF y los coeficientes se muestran por separado. Los resultados ayudan a priorizar preguntas de diagnóstico por área y a planificar el seguimiento de las medidas acordadas.
