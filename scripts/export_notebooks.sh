#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-docs/notebooks_html}"
mkdir -p "$OUT_DIR"

for notebook in 01_EDA_Attrition 02_Modelado_Baseline 03_Modelado_DL 05_Dashboard_KPIs 99_Informe_Final; do
  jupyter nbconvert --to html --output-dir "$OUT_DIR" "notebooks/$notebook.ipynb"
done

echo "HTML exportados en $OUT_DIR"
