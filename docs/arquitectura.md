# Arquitectura del proyecto

## Servicios

- **Jupyter:** entorno de trabajo con PySpark y scikit-learn. El puerto `8889` se publica solo en `127.0.0.1`; Jupyter genera un token al iniciar.
- **Spark Master:** expone el RPC `7077` y la interfaz `8080` en la máquina local. El ETL utiliza por defecto el modo local de Spark (`local[*]`) desde Jupyter; Compose incluye el servicio Master y se puede ampliar con workers para procesamiento distribuido.
- **PostgreSQL:** almacena las tablas de características y resultados en el puerto local `5442`, enlazado a `127.0.0.1`.

## Rutas compartidas

Jupyter monta las siguientes rutas del repositorio:

- `./scripts` → `/scripts`
- `./data` → `/data`
- `./output` → `/output`
- `./notebooks` → `/home/jovyan/work`
- ./docs/powerbi → /docs/powerbi

PostgreSQL conserva sus datos en `./postgres`. Al añadir workers de Spark, configura en ellos las rutas de datos que necesite el proceso distribuido.

## Arranque y comprobación

```bash
docker compose up -d --build
docker compose logs jupyter
```

Abre `http://localhost:8889/lab` e introduce el token que aparece en los registros. Para revisar PySpark en el modo local configurado por defecto, ejecuta en un notebook:

```python
from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .master("local[*]")
         .appName("revision-datos")
         .getOrCreate())
spark.read.csv("/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
               header=True, inferSchema=True).limit(5).show()
```

El proceso ETL y la secuencia completa están descritos en el [README](../README.md).\n\nLa configuración sensible se toma de `.env` y se ejemplifica en `.env.example`. El esquema inicial de PostgreSQL está en `postgres/init/001_schema.sql`.
