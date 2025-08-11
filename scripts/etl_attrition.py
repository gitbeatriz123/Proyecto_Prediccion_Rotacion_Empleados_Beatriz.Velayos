# -*- coding: utf-8 -*-
import argparse, json, os
from datetime import datetime, timezone
from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser("ETL Attrition + Encuesta (PySpark)")
    p.add_argument("--input1", required=True, help="CSV HR Attrition")
    p.add_argument("--input2", required=True, help="CSV Encuesta clima")
    p.add_argument("--key", default="EmployeeNumber")
    p.add_argument("--outdir", default="/data/processed/employee_attrition.parquet")
    p.add_argument("--spark-master", default=os.getenv("SPARK_MASTER", "local[*]"))
    return p.parse_args()

def spark_session(master):
    return (SparkSession.builder
            .master(master)
            .appName("etl-attrition")
            .config("spark.sql.execution.arrow.pyspark.enabled","true")
            .getOrCreate())

def trim_strings(df):
    exprs = [F.trim(F.col(c)).alias(c) if t == "string" else F.col(c).alias(c) for c,t in df.dtypes]
    return df.select(*exprs)

def main():
    args = parse_args()
    spark = spark_session(args.spark_master)

    # 1) Leer
    hr  = spark.read.csv(args.input1, header=True, inferSchema=True)
    enc = spark.read.csv(args.input2, header=True, inferSchema=True)

    # 2) Normalizar strings
    hr  = trim_strings(hr)
    enc = trim_strings(enc)

    # 3) Evitar colisiones de nombres en encuesta
    key = args.key
    for c in [c for c in enc.columns if c != key]:
        enc = enc.withColumnRenamed(c, f"survey_{c}")

    # 4) Join
    df = hr.join(enc, on=key, how="inner")

    # 5) Feature engineering (ejemplos)
    eps = F.lit(1e-6)
    df = (df
          .withColumn("attrition_label", F.when(F.col("Attrition") == "Yes", 1).otherwise(0))
          .withColumn("overtime_flag",   F.when(F.col("OverTime")  == "Yes", 1).otherwise(0))
          .withColumn("income_yearly",   F.col("MonthlyIncome") * 12)
          .withColumn("tenure_ratio",    F.col("YearsInCurrentRole") / (F.col("TotalWorkingYears") + eps))
         )

    # 6) Guardar en Parquet
    (df.repartition(1)  # desarrollo: un solo fichero por comodidad
       .write.mode("overwrite")
       .parquet(args.outdir))

    # 7) Métricas rápidas
    total = df.count()
    pos   = df.filter(F.col("attrition_label")==1).count()
    metrics = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "rows": total,
        "attrition_positive": pos,
        "attrition_rate": (pos/total if total else 0.0),
        "columns": df.columns,
        "outdir": args.outdir
    }
    os.makedirs("/output/metrics", exist_ok=True)
    with open("/output/metrics/etl_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ETL OK -> {args.outdir} | rows: {total} | attrition_rate: {metrics['attrition_rate']:.4f}")
    spark.stop()

if __name__ == "__main__":
    main()
