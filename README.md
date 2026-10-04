# Predicción de rotación de empleados

Proyecto de Big Data y analítica de Recursos Humanos para explorar factores asociados a la rotación de empleados y comparar modelos de clasificación. Incluye preparación de datos con PySpark, análisis en notebooks, modelos de scikit-learn y un dashboard de Power BI.

> Proyecto formativo y demostrativo. Los resultados no deben usarse para tomar decisiones sobre empleados reales.

## Qué incluye

- Control de calidad, integración de datos y ETL con PySpark.
- Análisis exploratorio y notebooks de modelado.
- Comparación de Regresión Logística, Random Forest y MLP.
- Persistencia de resultados en PostgreSQL y artefactos de modelos.
- Dashboard de Power BI con indicadores, segmentos y factores asociados al riesgo.

**Tecnologías:** Python, PySpark, scikit-learn, Jupyter, Docker Compose, PostgreSQL y Power BI.

## Dashboard de Power BI

![Resumen de KPIs del dashboard](docs/powerbi/01_Resumen_KPIs.png)

También puedes consultar las [cinco capturas del dashboard](docs/powerbi/).

## Entregables

- [Dashboard Power BI (PBIX)](bi/Proyecto_Beatriz.pbix)
- [Presentación final (PPTX)](docs/Presentacionfinal_Beatriz_Velayos.pptx)
- [Memoria del proyecto (PDF)](docs/Memoria_Proyecto_Beatriz.pdf) · [versión HTML](docs/Memoria_Proyecto_Beatriz.html)
- Informes HTML: [EDA](docs/01_EDA_Attrition.html), [modelado baseline](docs/02_Modelado_Baseline.html), [dashboard y KPIs](docs/05_Dashboard_KPIs.html) y [informe final](docs/99_Informe_Final.html)
- [Documentación de arquitectura](docs/arquitectura.md)

## Ejecución local

Requisitos: Docker con Docker Compose y espacio suficiente para las imágenes y dependencias.

1. Clona el repositorio y entra en su carpeta.
2. Construye y arranca los servicios:

   ```bash
   docker compose up -d --build
   ```

3. Consulta los registros para obtener el enlace y el token temporal que genera Jupyter:

   ```bash
   docker compose logs jupyter
   ```

4. Abre `http://localhost:8889/lab` y pega el token solicitado. El token se genera al iniciar Jupyter; no lo compartas ni lo guardes en el repositorio.

5. Para preparar el conjunto Parquet que utilizan los notebooks, ejecuta el ETL:

   ```bash
   docker compose run --rm jupyter python /scripts/etl_attrition.py \
     --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
     --input2 /data/raw/encuesta_clima.csv \
     --outdir /data/processed/employee_attrition.parquet
   ```

6. Abre los notebooks en Jupyter y ejecútalos en este orden:
   `01_EDA_Attrition`, `02_Modelado_Baseline`, `03_Modelado_DL`, `05_Dashboard_KPIs` y `99_Informe_Final`.

Para detener los servicios:

```bash
docker compose down
```

## Datos y evaluación

Los CSV de `data/raw/` contienen atributos de empleados y respuestas de una encuesta de clima, vinculados mediante `EmployeeNumber`. La encuesta se descargó de una fuente pública, pero no se ha recuperado su referencia exacta ni verificado su licencia de redistribución. Antes de compartir el proyecto, añade la fuente y sus condiciones de uso; si no puedes confirmarlas, retira ese CSV del repositorio y documenta cómo obtenerlo. No incluyas datos reales o confidenciales de una empresa.

Los resultados guardados en [`output/metrics/model_compare.json`](output/metrics/model_compare.json) proceden de más de un protocolo y no deben compararse como si fueran una única evaluación. Los resultados iniciales se calcularon con una partición estratificada de entrenamiento y prueba. Las entradas `LogReg` y `RF` del mismo archivo se calcularon con predicciones *out-of-fold* de validación cruzada estratificada de cinco particiones.

Los valores `f1_opt` y `thr_opt` corresponden a umbrales seleccionados para maximizar F1 usando esas mismas etiquetas de evaluación. Son resultados exploratorios, no una estimación independiente del rendimiento final. La Accuracy, por sí sola, tampoco describe adecuadamente el rendimiento en un problema con clases desbalanceadas.

## Estructura principal

```text
bi/                 Dashboard de Power BI
data/raw/           Datos de entrada
docs/               Memoria, presentación, informes y capturas
notebooks/          Análisis, modelado e informe
output/metrics/     Métricas versionadas
output/models/      Modelos versionados
scripts/            ETL, preprocesamiento y entrenamiento
docker-compose.yml  Servicios locales
```

Algunos artefactos de `output/` están versionados como referencia; los nuevos resultados generados localmente se excluyen mediante `.gitignore`.

## Uso responsable

Este trabajo es un ejercicio analítico y no valida decisiones de selección, promoción o retención de personas. Un modelo de rotación puede reflejar sesgos de los datos y requiere revisión de privacidad, equidad y contexto antes de cualquier uso real.

**Autora:** Beatriz Velayos · Proyecto formativo de Big Data, Machine Learning e Inteligencia Artificial.
