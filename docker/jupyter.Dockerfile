FROM jupyter/pyspark-notebook:latest

COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-ml.txt
