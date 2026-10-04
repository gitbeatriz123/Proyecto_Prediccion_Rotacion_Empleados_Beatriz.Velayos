# Predicción de rotación de empleados

Proyecto educativo de Big Data y analítica de Recursos Humanos desarrollado en un bootcamp. Integra preparación de datos con PySpark, análisis en Jupyter, modelos de clasificación con scikit-learn y un prototipo de dashboard en Power BI.

> **Demostración académica:** utiliza datos de ejemplo y resultados exploratorios. No es un sistema validado para evaluar, seleccionar, retener ni gestionar empleados. No debe ejecutarse con datos de la empresa ni usarse para tomar decisiones sobre personas.

## Qué incluye

- Comprobaciones de calidad y ETL con PySpark.
- Análisis exploratorio y notebooks reproducibles.
- Comparación exploratoria de Regresión Logística, Random Forest y MLP.
- Persistencia de artefactos y resultados en PostgreSQL/CSV.
- Un prototipo de Power BI y gráficos de referencia recreados a partir de artefactos versionados.

**Tecnologías:** Python, PySpark, scikit-learn, Jupyter, Docker Compose, PostgreSQL y Power BI.

## Dashboard

![Resumen de KPIs del prototipo](docs/powerbi/01_Resumen_KPIs.png)

Las imágenes de [`docs/powerbi/`](docs/powerbi/) son gráficos de referencia recreados con datos y resultados versionados; no son capturas verificadas de Power BI. El PBIX se incluye como prototipo y conviene abrirlo en Power BI para revisar sus visualizaciones antes de interpretarlo.

## Entregables

- [Prototipo Power BI (PBIX)](bi/Proyecto_Beatriz.pbix)
- [Presentación educativa (PPTX)](docs/Presentacionfinal_Beatriz_Velayos.pptx)
- [Memoria del proyecto (PDF)](docs/Memoria_Proyecto_Beatriz.pdf) · [versión HTML](docs/Memoria_Proyecto_Beatriz.html)
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
5. Genera el Parquet que consumen los notebooks:

   Primero, si quieres regenerar la encuesta sintética con la semilla publicada:

   ```bash
   docker compose run --rm jupyter python /scripts/generate_demo_survey.py \\
     --input /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \\
     --output /data/raw/encuesta_clima.csv --seed 42
   ```

   Este fichero contiene respuestas aleatorias sintéticas; no procede de una plantilla ni de una encuesta real.

   ```bash
   docker compose run --rm jupyter python /scripts/etl_attrition.py \
     --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --input2 /data/raw/encuesta_clima.csv \
     --outdir /data/processed/employee_attrition.parquet
   ```

6. Abre y ejecuta los notebooks en este orden: `01_EDA_Attrition`, `02_Modelado_Baseline`, `03_Modelado_DL`, `05_Dashboard_KPIs` y `99_Informe_Final`.

   Para recalcular la comparación homogénea de los tres modelos después del ETL:

   ```bash
   docker compose run --rm jupyter python /scripts/train_ml.py \\
     --input /data/processed/employee_attrition.parquet --model all
   ```

   La evaluación usa predicciones out-of-fold de validación cruzada estratificada de cinco particiones. El F1 y el umbral se seleccionan sobre esas mismas predicciones, así que son exploratorios y no una estimación independiente del rendimiento futuro.

Para detener los servicios: `docker compose down`. El ETL usa Spark en modo local por defecto; el servicio Spark Master no tiene un Worker configurado.

## Datos, resultados y límites

El archivo principal `WA_Fn-UseC_-HR-Employee-Attrition.csv` es el conjunto de ejemplo de rotación que se suele distribuir con el nombre IBM HR Analytics Employee Attrition & Performance. IBM usa ese archivo en un tutorial de clasificación, aunque eso no verifica por sí solo la licencia de esta copia. [Referencia del tutorial de IBM](https://developer.ibm.com/caas-storage/skillscollection/dna/live/innovator-predict-employee-turnover-using-ibm-watson-studio/en/_attachments/Build-train-and-evaluate-Machine-Learning-models.pdf).

`encuesta_clima.csv` se genera para este proyecto con `scripts/generate_demo_survey.py`: las cinco escalas se asignan aleatoriamente entre 1 y 5 con semilla fija 42, sin consultar la etiqueta de rotación. Son datos sintéticos reproducibles, no respuestas de personas ni mediciones de una plantilla o empresa. El CSV principal corresponde al conjunto de ejemplo conocido como IBM HR Analytics Employee Attrition & Performance. La referencia del tutorial de IBM identifica un conjunto con ese nombre, pero no verifica la licencia de redistribución de esta copia.

`output/metrics/model_compare.json` se regeneró con el mismo protocolo para los tres modelos: predicciones *out-of-fold* de validación cruzada estratificada de cinco particiones. Las métricas actuales son exploratorias. Para LogReg, Random Forest y MLP, respectivamente, ROC-AUC = 0,819 / 0,810 / 0,768; PR-AUC = 0,552 / 0,532 / 0,484; F1 optimizado = 0,533 / 0,544 / 0,470. El umbral y el F1 se eligen sobre las mismas predicciones OOF, por lo que no son una estimación independiente del rendimiento futuro.

No se ha realizado validación temporal, calibración independiente, análisis formal de equidad ni validación para decisiones laborales. El rendimiento en este dataset de ejemplo no predice el de otra organización. Las métricas versionadas se regeneraron con el código actual. El F1 y el umbral se optimizan sobre las mismas predicciones OOF, por lo que no son una estimación independiente del rendimiento futuro.

## Estructura

```text
bi/                 Prototipo Power BI
data/raw/           Datos de entrada del ejercicio
docs/               Memoria, presentación, informes y capturas
notebooks/          Análisis, modelado e informe
output/             Métricas, modelos y exportaciones de referencia
scripts/            ETL, comprobaciones y entrenamiento
docker-compose.yml  Servicios locales
```

Los artefactos de `output/` versionados son resultados de referencia reproducibles; vuelve a generarlos si cambias los datos, las dependencias o el código.

## Licencia y reutilización

El repositorio no declara una licencia general de reutilización. La encuesta sintética se puede volver a generar desde el script. Verifica las condiciones del conjunto HR de ejemplo antes de redistribuirlo por separado.

## Uso responsable

Es un proyecto formativo. Los datos no representan a la empresa de la autora. No introduzcas datos personales, confidenciales o de empleados reales. Un modelo de rotación puede reproducir sesgos y no permite concluir causalidad; cualquier uso laboral requeriría revisión jurídica, de privacidad, equidad y contexto, además de validación independiente.

**Autora:** Beatriz Velayos · Proyecto formativo de Big Data, Machine Learning e Inteligencia Artificial.
