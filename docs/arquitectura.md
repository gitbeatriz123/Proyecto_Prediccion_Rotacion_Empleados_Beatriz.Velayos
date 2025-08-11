# Arquitectura — Entorno local (pasos 0 y 1)

## Componentes
- **Spark Master / Worker** (bitnami/spark): procesamiento distribuido.
- **Jupyter (pyspark-notebook)**: desarrollo interactivo y pruebas.
- **PostgreSQL (opcional)**: persistencia de métricas/predicciones para BI.

## Puertos
- Spark Master UI: `http://localhost:8080`
- Spark Worker UI: `http://localhost:8081`
- Jupyter: `http://localhost:8888`
- PostgreSQL: `localhost:5432`

## Volúmenes compartidos
- `./data` → `/data` (todos los servicios)
- `./notebooks` → `/home/jovyan/work` (Jupyter)
- `./scripts` → `/scripts` (todos los servicios)
- `./output` → `/output` (Postgres y exportaciones)

> Nota: Leer ficheros siempre desde **`/data/...`** para que Spark en los workers pueda acceder a la misma ruta.

## Diagrama (simplificado)

           +-------------------------+
           |       Jupyter           |
           |  (pyspark-notebook)     |
           |  Driver PySpark         |
           +------------+------------+
                        |
                        | spark://spark-master:7077
                        v
           +-------------------------+
           |      Spark Master       |
           +------------+------------+
                        |
                        v
           +-------------------------+
           |      Spark Worker       |
           +-------------------------+

           Volúmenes compartidos: /data, /scripts, /output

## Prueba rápida
1) `docker compose up -d`
2) En Jupyter, ejecuta:
```python
from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .master("spark://spark-master:7077")
         .appName("smoke-test")
         .getOrCreate())
spark.read.csv("/data/raw", header=True, inferSchema=True).limit(5).show()
```
