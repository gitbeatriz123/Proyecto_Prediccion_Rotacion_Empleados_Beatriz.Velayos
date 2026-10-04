#!/usr/bin/env python3
"""Genera las cinco figuras de resumen a partir de las métricas y predicciones versionadas."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent.parent
COLORS = {
    "navy": "#17324F",
    "teal": "#176B78",
    "blue": "#2D7798",
    "mint": "#60BEB5",
    "gold": "#D7A21A",
    "coral": "#D97867",
    "muted": "#53677B",
    "grid": "#DCE5EB",
    "positive": "#C95F54",
    "negative": "#277F96",
}


def fmt_number(value, decimals=1):
    return f"{value:.{decimals}f}".replace(".", ",")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, path, caption):
    fig.text(0.5, 0.025, caption, ha="center", va="bottom",
             fontsize=8.5, color=COLORS["muted"], wrap=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate(metrics_path, predictions_path, effects_path, output_dir):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = read_csv(predictions_path)
    effects = read_csv(effects_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelcolor": COLORS["muted"],
        "figure.facecolor": "white",
    })

    # 1. Modelo: comparación común de ROC-AUC, PR-AUC y F1.
    model_keys = ["LogReg", "RF", "MLP"]
    model_names = ["Regresión logística", "Random Forest", "MLP"]
    metric_defs = [
        ("roc_auc", "ROC-AUC", COLORS["blue"]),
        ("pr_auc", "PR-AUC", COLORS["mint"]),
        ("f1_opt", "F1 comparativo", COLORS["gold"]),
    ]
    x = np.arange(len(model_keys))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for j, (key, label, color) in enumerate(metric_defs):
        vals = [metrics[m][key] for m in model_keys]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label, color=color, zorder=3)
        ax.bar_label(bars, labels=[fmt_number(value, 3) for value in vals], padding=3, fontsize=8, color=COLORS["navy"])
    ax.set_title("Comparación de modelos de clasificación", color=COLORS["navy"], pad=14)
    ax.set_ylabel("Valor de la métrica")
    ax.set_xticks(x, model_names)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt_number(value, 1)))
    style_axes(ax)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.23, top=0.84)
    finish(fig, output_dir / "01_Resumen_KPIs.png",
           "Regresión Logística lidera ROC-AUC y PR-AUC; Random Forest logra el F1 más alto. Validación cruzada OOF: cinco particiones.")

    # Aggregate OOF scores, observed exits, departments and overtime directly from one file.
    scores = []
    by_department = defaultdict(lambda: [0, 0, 0])
    by_overtime = defaultdict(lambda: [0, 0, 0])
    threshold = float(metrics["LogReg"]["thr_opt"])
    flags = defaultdict(int)
    for row in predictions:
        score = float(row["proba"])
        y = int(row["attrition_label"])
        department = row["department"]
        overtime = row["overtime"]
        scores.append(score)
        by_department[department][0] += 1
        by_department[department][1] += y
        by_overtime[overtime][0] += 1
        by_overtime[overtime][1] += y
        if score >= threshold:
            flags[department] += 1

    # 2. Out-of-fold score distribution with counts from the saved prediction file.
    bin_counts = [0] * 10
    for value in scores:
        bin_counts[min(int(value * 10), 9)] += 1
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar(edges[:-1], bin_counts, width=0.082, align="edge",
                  color=COLORS["blue"], edgecolor="white", linewidth=1)
    ax.bar_label(bars, padding=3, fontsize=8, color=COLORS["navy"])
    ax.set_title("Distribución de puntuaciones OOF · Regresión Logística",
                 color=COLORS["navy"], pad=14)
    ax.set_xlabel("Intervalos de puntuación")
    ax.set_ylabel("Número de registros")
    ax.set_xticks(edges, [f"{edge:.1f}".replace(".", ",") for edge in edges], fontsize=8)
    ax.set_xlim(0, 1)
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.23, top=0.84)
    below_03 = sum(bin_counts[:3])
    pct = below_03 / len(scores) * 100
    finish(fig, output_dir / "02_Distribucion_Puntuaciones.png",
           f"{below_03} de {len(scores)} puntuaciones ({fmt_number(pct)} %) quedan por debajo de 0,30.")

    # 3. Department-level observed rates, counts and OOF prioritization signals.
    dept_names = {
        "Sales": "Ventas",
        "Human Resources": "Recursos Humanos",
        "Research & Development": "Investigación y Desarrollo",
    }
    dept_rows = []
    for name, values in by_department.items():
        n, exits, _ = values
        dept_rows.append((name, n, exits, exits / n, flags[name]))
    dept_rows.sort(key=lambda row: row[3], reverse=True)
    labels = [dept_names.get(row[0], row[0]) for row in dept_rows]
    rates = [row[3] * 100 for row in dept_rows]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.barh(labels[::-1], rates[::-1], color=[COLORS["coral"], COLORS["mint"], COLORS["mint"]][::-1], height=0.58)
    for bar, row in zip(bars, dept_rows[::-1]):
        ax.text(bar.get_width() + 0.35, bar.get_y() + bar.get_height() / 2,
                f"{fmt_number(bar.get_width())} %  ·  {row[2]}/{row[1]} salidas  ·  {row[4]} señales",
                va="center", ha="left", fontsize=8.5, color=COLORS["navy"])
    ax.set_title("Tasa observada de rotación por departamento", color=COLORS["navy"], pad=14)
    ax.set_xlabel("Tasa observada (%)")
    ax.set_xlim(0, max(rates) + 14)
    style_axes(ax)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.20, top=0.84)
    finish(fig, output_dir / "03_Departamentos_Tasa_Rotacion.png",
           "Señales OOF al umbral comparativo 0,732. Ventas lidera la tasa e I+D concentra el mayor número de señales.")

    # 4. Overtime: compare the observed rates and denominators.
    overtime_names = {"No": "Sin horas extra", "Yes": "Con horas extra"}
    order = ["No", "Yes"]
    rates = [by_overtime[name][1] / by_overtime[name][0] * 100 for name in order]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar([overtime_names[name] for name in order], rates,
                  color=[COLORS["blue"], COLORS["coral"]], width=0.52, zorder=3)
    for bar, name, rate in zip(bars, order, rates):
        n, exits, _ = by_overtime[name]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{fmt_number(rate)} %  ·  {exits}/{n}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLORS["navy"])
    ax.set_title("Tasa observada según horas extra", color=COLORS["navy"], pad=14)
    ax.set_ylabel("Tasa observada (%)")
    ax.set_ylim(0, max(rates) + 10)
    style_axes(ax)
    fig.subplots_adjust(bottom=0.23, top=0.84)
    finish(fig, output_dir / "04_Tasa_Rotacion_Horas_Extra.png",
           "La diferencia entre grupos orienta una revisión de turnos, planificación y distribución de carga.")

    # 5. Translate the leading model features for the business audience.
    feature_names = {
        "JobRole_Research Director": "Director/a de investigación",
        "JobRole_Sales Representative": "Representante de ventas",
        "BusinessTravel_Travel_Frequently": "Viajes frecuentes",
        "BusinessTravel_Non-Travel": "Sin viajes de trabajo",
        "JobRole_Laboratory Technician": "Técnico/a de laboratorio",
        "JobRole_Human Resources": "Puesto en Recursos Humanos",
        "YearsAtCompany": "Años en la empresa",
        "MaritalStatus_Single": "Estado civil: soltero/a",
        "EducationField_Technical Degree": "Formación técnica",
        "JobRole_Healthcare Representative": "Representante sanitario",
    }
    top = sorted(effects, key=lambda row: abs(float(row["effect"])), reverse=True)[:10]
    top = [(feature_names.get(row["feature"], row["feature"]), float(row["effect"])) for row in top]
    top.sort(key=lambda row: row[1])
    labels = [row[0] for row in top]
    values = [row[1] for row in top]
    colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in values]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    bars = ax.barh(labels, values, color=colors, height=0.66, zorder=3)
    ax.axvline(0, color=COLORS["muted"], linewidth=0.8)
    for bar, value in zip(bars, values):
        pad = 0.025 if value >= 0 else -0.025
        ax.text(value + pad, bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}".replace(".", ","),
                va="center", ha="left" if value >= 0 else "right",
                fontsize=8.5, color=COLORS["navy"])
    ax.set_title("Factores asociados · Regresión Logística", color=COLORS["navy"], pad=14)
    ax.set_xlabel("Coeficiente del modelo")
    ax.set_xlim(min(values) - 0.25, max(values) + 0.35)
    style_axes(ax)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.17, top=0.88)
    finish(fig, output_dir / "05_Factores_Asociados_Modelo.png",
           "Representantes de ventas y viajes frecuentes destacan entre las asociaciones positivas de mayor magnitud.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "output/metrics/model_compare.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "output/bi/predictions_logreg.csv")
    parser.add_argument("--effects", type=Path, default=ROOT / "output/bi/feature_effects.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/powerbi")
    args = parser.parse_args()
    generate(args.metrics, args.predictions, args.effects, args.output_dir)


if __name__ == "__main__":
    main()
