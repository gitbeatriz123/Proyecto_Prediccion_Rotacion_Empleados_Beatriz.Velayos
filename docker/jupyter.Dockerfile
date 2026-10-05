FROM jupyter/pyspark-notebook:spark-3.5.6

COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-ml.txt
