# -*- coding: utf-8 -*-
# Script de validación de calidad de datos para los pasos 0 y 1.
# Uso:
#   python /scripts/check_data_quality.py \
#     --input1 /data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
#     --input2 /data/raw/encuesta_clima.csv \
#     --key EmployeeNumber \
#     --max-null-frac 0.25

import argparse
import json
import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser(description="Validación de calidad de datos (PySpark)")
    p.add_argument("--input1", required=True, help="Ruta CSV 1 (HR Attrition)")
    p.add_argument("--input2", required=True, help="Ruta CSV 2 (Encuesta clima)")
    p.add_argument("--key", required=True, help="Clave de unión (ej. EmployeeNumber)")
    p.add_argument("--max-null-frac", type=float, default=0.25, help="Umbral máximo de fracción de nulos por columna")
    p.add_argument("--spark-master", default=os.getenv("SPARK_MASTER", "spark://spark-master:7077"),
                   help="URL del Spark master (por defecto lee SPARK_MASTER o spark://spark-master:7077)")
    return p.parse_args()

def build_spark(master_url: str) -> SparkSession:
    return (SparkSession.builder
            .master(master_url)
            .appName("quality-check")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate())

def fraction_nulls(df):
    total = df.count()
    stats = {}
    for c in df.columns:
        n_null = df.filter(F.col(c).isNull() | (F.trim(F.col(c)) == "")).count()
        stats[c] = {"nulls": int(n_null), "frac": (float(n_null) / float(total)) if total > 0 else 0.0}
    return stats

def domain_check_attrition(df, col="Attrition"):
    if col not in df.columns:
        return {"present": False, "valid": None, "invalid_values": []}
    allowed = {"Yes", "No"}
    values = [r[0] for r in df.select(col).distinct().collect() if r[0] is not None]
    invalid = [v for v in values if v not in allowed]
    return {"present": True, "valid": len(invalid) == 0, "invalid_values": invalid, "values": values}

def main():
    args = parse_args()
    spark = build_spark(args.spark_master)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {"input1": args.input1, "input2": args.input2},
        "key": args.key,
        "max_null_frac": args.max_null_frac,
        "checks": {}
    }
    exit_code = 0

    # Lectura de CSV
    try:
        df1 = spark.read.csv(args.input1, header=True, inferSchema=True)
        df2 = spark.read.csv(args.input2, header=True, inferSchema=True)
    except Exception as e:
        print(f"[ERROR] No se pudieron leer los CSV: {e}", file=sys.stderr)
        sys.exit(2)

    # Check: columna clave no nula
    if args.key not in df1.columns or args.key not in df2.columns:
        print(f"[ERROR] La clave {args.key} no existe en ambos datasets", file=sys.stderr)
        sys.exit(2)

    null_key_1 = df1.filter(F.col(args.key).isNull()).count()
    null_key_2 = df2.filter(F.col(args.key).isNull()).count()
    report["checks"]["null_key"] = {"input1_nulls": int(null_key_1), "input2_nulls": int(null_key_2)}
    if null_key_1 > 0 or null_key_2 > 0:
        exit_code = 1

    # Check: duplicados por clave
    dup1 = df1.groupBy(args.key).count().filter(F.col("count") > 1).count()
    dup2 = df2.groupBy(args.key).count().filter(F.col("count") > 1).count()
    report["checks"]["duplicates"] = {"input1_duplicates": int(dup1), "input2_duplicates": int(dup2)}
    if dup1 > 0 or dup2 > 0:
        exit_code = 1

    # Check: fracción de nulos por columna
    nulls1 = fraction_nulls(df1)
    nulls2 = fraction_nulls(df2)
    report["checks"]["nulls_fraction"] = {"input1": nulls1, "input2": nulls2}

    # Señala columnas por encima del umbral
    over1 = {k: v for k, v in nulls1.items() if v["frac"] > args.max_null_frac}
    over2 = {k: v for k, v in nulls2.items() if v["frac"] > args.max_null_frac}
    report["checks"]["nulls_over_threshold"] = {"input1": over1, "input2": over2}
    if len(over1) > 0 or len(over2) > 0:
        exit_code = 1

    # Check de dominio (Attrition Yes/No) en input1
    report["checks"]["attrition_domain_input1"] = domain_check_attrition(df1, "Attrition")

    # Resumen
    report["summary"] = {
        "status": "PASS" if exit_code == 0 else "FAIL",
        "hints": [
            "Asegura que la columna clave no tiene nulos en ambos datasets.",
            "Elimina duplicados por clave antes del join.",
            f"Imputa o elimina columnas con fracción de nulos > {args.max_null_frac}.",
            "Homogeneiza dominios categóricos (ej. Attrition en {Yes, No})."
        ]
    }

    # Guardar informe
    out_dir = "/output/metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quality_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
