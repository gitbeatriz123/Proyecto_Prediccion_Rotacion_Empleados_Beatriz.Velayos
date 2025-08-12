# scripts/etl_attrition.py
# -*- coding: utf-8 -*-
"""
ETL de rotación de empleados:
- Lee CSV principal (HR) y encuesta de clima
- Limpia y tipifica
- Une por EmployeeNumber
- Crea label y variables derivadas
- (CRÍTICO) Elimina la columna objetivo 'Attrition' para evitar data leakage
- Escribe en Parquet (Snappy) y guarda métricas JSON
"""

import os
import json
from datetime import datetime, timezone
import argparse

from pyspark.sql import SparkSession, functions as F, types as T


# ----------------------------
# Argumentos CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input1", required=True,
                   help="CSV principal HR (WA_Fn-UseC_-HR-Employee-Attrition.csv)")
    p.add_argument("--input2", required=True,
                   help="CSV encuesta clima (encuesta_clima.csv)")
    p.add_argument("--key", default="EmployeeNumber",
                   help="Columna clave para el join (por defecto EmployeeNumber)")
    p.add_argument("--outdir", required=True,
                   help="Directorio de salida Parquet (p.ej. /data/processed/employee_attrition.parquet)")
    p.add_argument("--spark-master", default=os.getenv("SPARK_MASTER", "local[*]"),
                   help="Master de Spark (por defecto local[*])")
    return p.parse_args()


# ----------------------------
# Spark session
# ----------------------------
def spark_session(master: str) -> SparkSession:
    spark = (
        SparkSession
        .builder
        .appName("etl-attrition")
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ----------------------------
# Utilidades
# ----------------------------
def trim_all_strings(df):
    """Aplica trim a todas las columnas string."""
    exprs = [
        F.trim(F.col(c)).alias(c) if t == "string" else F.col(c).alias(c)
        for c, t in df.dtypes
    ]
    return df.select(*exprs)


def rename_if_needed(df, old, new):
    return df.withColumnRenamed(old, new) if old in df.columns else df


def prefix_if_missing(df, cols, prefix):
    out = df
    for c in cols:
        if c in out.columns and not c.startswith(prefix):
            out = out.withColumnRenamed(c, f"{prefix}{c}")
    return out


# ----------------------------
# Lectura, limpieza, unión
# ----------------------------
def read_hr_csv(spark, path):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .csv(path)
    )
    df = trim_all_strings(df)
    return df


def read_survey_csv(spark, path, key):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )
    df = trim_all_strings(df)

    # Normaliza nombre de clave (por si llega como employeeNumber o similar)
    for k in df.columns:
        if k.lower() == key.lower() and k != key:
            df = df.withColumnRenamed(k, key)

    # Fuerza enteros en encuesta
    int_cols = [
        "Engagement",
        "Satisfaction",
        "WorkLifeBalanceSurvey",
        "ManagerRelationship",
        "RemoteWorkSatisfaction",
    ]
    for c in int_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(T.IntegerType()))

    # Prefijo claro
    df = prefix_if_missing(df, int_cols, "survey_")
    return df


# ----------------------------
# Feature engineering
# ----------------------------
def feature_engineering(df):
    eps = F.lit(1e-9)

    # etiqueta binaria a partir de Attrition (Yes/No)
    if "Attrition" in df.columns:
        df = df.withColumn(
            "attrition_label",
            (F.col("Attrition") == F.lit("Yes")).cast(T.IntegerType())
        )
        # (CRÍTICO) Evitar fuga: quitar la columna objetivo textual
        df = df.drop("Attrition")

    # Flags y ratios
    if "OverTime" in df.columns:
        df = df.withColumn("overtime_flag", (F.col("OverTime") == F.lit("Yes")).cast(T.IntegerType()))

    if "MonthlyIncome" in df.columns:
        df = df.withColumn("income_yearly", (F.col("MonthlyIncome").cast(T.DoubleType()) * F.lit(12.0)))

    # Tenure ratio: YearsAtCompany / (TotalWorkingYears + eps)
    if all(c in df.columns for c in ["YearsAtCompany", "TotalWorkingYears"]):
        df = df.withColumn(
            "tenure_ratio",
            (F.col("YearsAtCompany").cast(T.DoubleType()) / (F.col("TotalWorkingYears").cast(T.DoubleType()) + eps))
        )

    return df


# ----------------------------
# Escritura y métricas
# ----------------------------
def write_parquet(df, outdir):
    (
        df
        .coalesce(1)  # 1 fichero para facilitar el consumo aguas abajo / demo
        .write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(outdir)
    )


def save_metrics(df, outdir):
    out_metrics = "/output/metrics/etl_metrics.json"
    rows = df.count()
    pos = df.filter(F.col("attrition_label") == 1).count() if "attrition_label" in df.columns else None
    rate = (pos / rows) if (pos is not None and rows > 0) else None

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "attrition_positive": pos,
        "attrition_rate": rate,
        "columns": df.columns,
        "outdir": outdir,
    }
    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    spark = spark_session(args.spark_master)

    # Lee datasets
    hr = read_hr_csv(spark, args.input1)
    survey = read_survey_csv(spark, args.input2, args.key)

    # Dedup en la clave
    hr = hr.dropDuplicates([args.key])
    survey = survey.dropDuplicates([args.key])

    # Join left (hr como principal)
    df = (
        hr.alias("hr")
        .join(survey.alias("sv"), on=args.key, how="left")
    )

    # Features
    df = feature_engineering(df)

    # Escritura
    write_parquet(df, args.outdir)

    # Métricas
    save_metrics(df, args.outdir)

    print(f"ETL OK -> {args.outdir} | rows: {df.count()} | "
          f"attrition_label in cols: {'attrition_label' in df.columns}")
    spark.stop()


if __name__ == "__main__":
    main()

