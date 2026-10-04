# Arquitectura del proyecto local

## Servicios definidos

El archivo `docker-compose.yml` configura tres servicios:

- **Jupyter:** entorno de trabajo con PySpark y scikit-learn. El puerto `8889` del equipo se publica solo en `127.0.0.1` y Jupyter genera un token al iniciar.
- **Spark Master:** publica el RPC `7077` y la interfaz `8080` solo en `127.0.0.1`. No hay un servicio Spark Worker definido en Compose. Los notebooks y el ETL usan `local[*]` por defecto; el Master no participa en ese modo local.
- **PostgreSQL:** base local de demostración, puerto `5442` del equipo enlazado solo a `127.0.0.1`. Las credenciales incluidas en Compose son exclusivamente de ejemplo y no deben reutilizarse en despliegues.

## Rutas compartidas

Solo el contenedor Jupyter monta estas rutas:

- `./scripts` → `/scripts`
- `./data` → `/data`
- `./output` → `/output`
- `./notebooks` → `/home/jovyan/work`

PostgreSQL persiste sus datos en `./postgres`. Para que Spark distribuido acceda a los mismos archivos, habría que configurar y montar las rutas también en los workers; esa configuración no forma parte de este prototipo.

## Arranque y comprobación local

```bash
docker compose up -d --build
docker compose logs jupyter
```

Abre `http://localhost:8889/lab` y usa el token de los registros. Para verificar PySpark en el modo configurado por defecto, ejecuta en un notebook:

```python
from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .master("local[*]")
         .appName("smoke-test")
         .getOrCreate())
spark.read.csv("/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
               header=True, inferSchema=True).limit(5).show()
```

El ETL y las instrucciones completas están en el [README](../README.md). Esta arquitectura sirve para aprendizaje local; no describe un clúster distribuido listo para producción.
