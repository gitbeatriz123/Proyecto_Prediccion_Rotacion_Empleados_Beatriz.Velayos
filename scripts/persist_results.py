import argparse, json, joblib, numpy as np, pandas as pd
from pathlib import Path

# pip install psycopg2-binary ya ejecutado en el contenedor jupyter
import psycopg2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)          # parquet con EmployeeNumber y attrition_label
    ap.add_argument("--model", required=True)          # ruta del pipeline .pkl
    ap.add_argument("--model_name", required=True)     # LogReg | RF | ...
    ap.add_argument("--metrics_json", required=True)   # /output/metrics/model_compare.json
    ap.add_argument("--pg_host", default="postgres")
    ap.add_argument("--pg_db",   default="postgres")
    ap.add_argument("--pg_user", default="postgres")
    ap.add_argument("--pg_pass", default="postgres")
    a = ap.parse_args()

    # Carga datos y modelo
    df = pd.read_parquet(a.input)
    y  = df["attrition_label"].astype(int).values
    X  = df.drop(columns=["attrition_label"])
    emp = df["EmployeeNumber"].values if "EmployeeNumber" in df.columns else np.arange(len(df))

    pipe  = joblib.load(a.model)
    probs = pipe.predict_proba(X)[:, 1]

    # Umbral desde JSON
    compare = json.loads(Path(a.metrics_json).read_text())
    m = compare.get(a.model_name, {})
    thr = float(m.get("thr_opt", 0.5))
    preds = (probs >= thr).astype(int)

    # Conexión
    conn = psycopg2.connect(host=a.pg_host, dbname=a.pg_db, user=a.pg_user, password=a.pg_pass)
    cur  = conn.cursor()

    # Guarda métricas (reemplaza entrada anterior del mismo modelo)
    cur.execute("DELETE FROM metrics WHERE model_name=%s", (a.model_name,))
    cur.execute("""
        INSERT INTO metrics(model_name, roc_auc, pr_auc, f1_opt, thr_opt)
        VALUES (%s,%s,%s,%s,%s)
    """, (a.model_name, m.get("roc_auc"), m.get("pr_auc"), m.get("f1_opt"), m.get("thr_opt")))
    conn.commit()

    # Reemplaza predicciones previas del mismo modelo para no duplicar
    cur.execute("DELETE FROM predictions WHERE model_name=%s", (a.model_name,))
    rows = list(zip(emp.tolist(), probs.tolist(), preds.tolist(), [a.model_name]*len(emp)))
    cur.executemany("""
        INSERT INTO predictions(employee_number, proba, pred, model_name)
        VALUES (%s,%s,%s,%s)
    """, rows)
    conn.commit()

    cur.close(); conn.close()
    print(f"Persistido en Postgres: {a.model_name} -> {len(rows)} predicciones (thr={thr:.3f})")

if __name__ == "__main__":
    main()
