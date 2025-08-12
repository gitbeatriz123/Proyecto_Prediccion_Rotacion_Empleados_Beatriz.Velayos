import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out",   required=True)
a = ap.parse_args()

df = pd.read_parquet(a.input)
X  = df.drop(columns=["attrition_label"])

# ✅ detectar numéricos y booleanos de forma robusta
num_cols = X.select_dtypes(include=[np.number, "Int64", "Float64", "boolean", "bool"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    [("num", StandardScaler(with_mean=False), num_cols),
     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="drop"
)

pre.fit(X)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"preprocessor": pre, "num_cols": num_cols, "cat_cols": cat_cols}, a.out)
print("OK ->", a.out, "| num:", len(num_cols), "cat:", len(cat_cols))
