FROM jupyter/pyspark-notebook:latest
RUN pip install --no-cache-dir psycopg2-binary
