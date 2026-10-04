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
- Informes HTML: [EDA](docs/01_EDA_Attrition.html), [modelado baseline](docs/02_Modelado_Baseline.html), [dashboard y KPIs](docs/05_Dashboard_KPIs.html) y [informe final](docs/99_Informe_Final.html)
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

   ```bash
   docker compose run --rm jupyter python /scripts/etl_attrition.py \
     --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --input2 /data/raw/encuesta_clima.csv \
     --outdir /data/processed/employee_attrition.parquet
   ```

6. Abre y ejecuta los notebooks en este orden: `01_EDA_Attrition`, `02_Modelado_Baseline`, `03_Modelado_DL`, `05_Dashboard_KPIs` y `99_Informe_Final`.

Para detener los servicios: `docker compose down`. El ETL usa Spark en modo local por defecto; el servicio Spark Master no tiene un Worker configurado.

## Datos, resultados y límites

El archivo principal `WA_Fn-UseC_-HR-Employee-Attrition.csv` es el conjunto de ejemplo de rotación que se suele distribuir con el nombre IBM HR Analytics Employee Attrition & Performance. IBM usa ese archivo en un tutorial de clasificación, aunque eso no verifica por sí solo la licencia de esta copia. [Referencia del tutorial de IBM](https://developer.ibm.com/caas-storage/skillscollection/dna/live/innovator-predict-employee-turnover-using-ibm-watson-studio/en/_attachments/Build-train-and-evaluate-Machine-Learning-models.pdf).

La referencia exacta y la licencia de redistribución de `encuesta_clima.csv` no se han podido recuperar. Sus cinco escalas presentan distribuciones casi uniformes; eso es compatible con datos de demostración generados, pero no permite confirmar su origen. Por tanto, el proyecto no afirma que sean respuestas reales ni que su procedencia esté verificada. No se deben interpretar como mediciones de una plantilla o empresa.

Las métricas de `output/metrics/model_compare.json` mezclan protocolos: las entradas `logreg`, `rf` y `mlp_sklearn` proceden de una partición estratificada de entrenamiento/prueba; `LogReg` y `RF` se calcularon con predicciones *out-of-fold* de validación cruzada estratificada de cinco particiones. No compares las cifras entre protocolos como si fueran una única evaluación. En las entradas OOF, el umbral y el F1 óptimos se seleccionaron sobre las mismas predicciones usadas para informar el resultado; son exploratorios y no una estimación independiente del rendimiento final.

No se ha realizado validación temporal, calibración independiente, análisis formal de equidad ni validación para decisiones laborales. El rendimiento en este dataset de ejemplo no predice el de otra organización. Los artefactos de métricas versionados son salidas históricas. Se corrigió en `scripts/train_ml.py` la alineación entre umbrales y puntos de la curva precisión-recall, pero no se pudo volver a ejecutar el entrenamiento en este entorno; regenera los resultados antes de citar los valores como definitivos.

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

Algunos artefactos de `output/` están versionados como referencia; los resultados nuevos generados localmente se excluyen mediante `.gitignore`.

## Licencia y reutilización

El repositorio no declara una licencia general de reutilización. La licencia de `encuesta_clima.csv` tampoco está verificada. Puedes compartir el enlace como muestra de un proyecto educativo, pero no presentes los datos como propios ni redistribuyas el fichero de encuesta por separado sin confirmar sus condiciones de uso.

## Uso responsable

Es un proyecto formativo. Los datos no representan a la empresa de la autora. No introduzcas datos personales, confidenciales o de empleados reales. Un modelo de rotación puede reproducir sesgos y no permite concluir causalidad; cualquier uso laboral requeriría revisión jurídica, de privacidad, equidad y contexto, además de validación independiente.

**Autora:** Beatriz Velayos · Proyecto formativo de Big Data, Machine Learning e Inteligencia Artificial.
