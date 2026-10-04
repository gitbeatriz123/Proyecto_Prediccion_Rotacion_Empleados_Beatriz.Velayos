"""Entrena y evalúa modelos con validación cruzada estratificada OOF."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDE = {"EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "Attrition", "attrition_label"}
MODELS = ("logreg", "rf", "mlp")


def evaluate_probs(y, probs):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_candidates = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.nanargmax(f1_candidates))
    pred_05 = (probs >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "f1_opt": float(f1_candidates[best_idx]),
        "thr_opt": float(thresholds[best_idx]),
        "accuracy_at_0_5": float(accuracy_score(y, pred_05)),
        "f1_at_0_5": float(f1_score(y, pred_05, zero_division=0)),
        "protocol": "stratified_5fold_out_of_fold",
    }


def make_pipeline(model_name, numeric, categorical):
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="drop")
    if model_name == "logreg":
        clf = LogisticRegression(solver="saga", penalty="l2", max_iter=3000,
                                 class_weight="balanced", random_state=42)
        label = "LogReg"
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=42, n_jobs=-1)
        label = "RF"
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=150,
                            early_stopping=True, random_state=42)
        label = "MLP"
    return Pipeline([("prep", pre), ("clf", clf)]), label


def save_curves(y, probs, name, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC")
    plt.tight_layout(); plt.savefig(plots_dir / f"{name.lower()}_roc.png", dpi=160); plt.close()
    precision, recall, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(5, 4)); plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} Precision–Recall")
    plt.tight_layout(); plt.savefig(plots_dir / f"{name.lower()}_pr.png", dpi=160); plt.close()


def train_one(model_name, df, model_dir, metric_dir, plot_dir, bi_dir):
    y = df["attrition_label"].astype(int).to_numpy()
    ids = df["EmployeeNumber"].to_numpy() if "EmployeeNumber" in df else np.arange(len(df))
    X = df.drop(columns=[c for c in df.columns if c in EXCLUDE])
    numeric = X.select_dtypes(include=[np.number, "Int64", "Float64", "boolean", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pipe, label = make_pipeline(model_name, numeric, categorical)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    metrics = evaluate_probs(y, probs)
    save_curves(y, probs, label, plot_dir)

    pipe.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True); metric_dir.mkdir(parents=True, exist_ok=True)
    bi_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / f"{model_name}_pipeline.pkl")
    (metric_dir / f"{model_name}_oof_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if model_name == "logreg":
        out_preds = pd.DataFrame({
            "employee_number": ids, "attrition_label": y,
            "proba": probs, "pred": (probs >= metrics["thr_opt"]).astype(int),
        })
        # Include only the aggregate chart descriptors required by the prototype.
        for source, target in (("Department", "department"), ("JobRole", "jobrole"),
                               ("Gender", "gender"), ("Age", "age"), ("MonthlyIncome", "monthlyincome"),
                               ("YearsAtCompany", "yearsatcompany"), ("OverTime", "overtime")):
            if source in df.columns:
                out_preds[target] = df[source].to_numpy()
        out_preds.to_csv(bi_dir / "predictions_logreg.csv", index=False)
        names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pipe.named_steps["clf"].coef_.ravel()
        effects = pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in names], "effect": coefs})
        effects.reindex(effects.effect.abs().sort_values(ascending=False).index).to_csv(
            metric_dir / "feature_effects.csv", index=False)

    if model_name == "rf":
        names = pipe.named_steps["prep"].get_feature_names_out()
        pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in names],
                      "importance": pipe.named_steps["clf"].feature_importances_}) \
          .sort_values("importance", ascending=False).to_csv(metric_dir / "feature_importance_rf.csv", index=False)
    print(f"{label}: ROC-AUC={metrics['roc_auc']:.3f}; PR-AUC={metrics['pr_auc']:.3f}; "
          f"F1*={metrics['f1_opt']:.3f} (umbral exploratorio {metrics['thr_opt']:.3f})")
    return label, metrics


def run(input_path, model_name, model_dir, metric_dir, plot_dir, bi_dir):
    df = pd.read_parquet(input_path)
    if "attrition_label" not in df:
        raise ValueError("Falta attrition_label en el Parquet.")
    models = MODELS if model_name == "all" else (model_name,)
    results = {}
    for current in models:
        label, metrics = train_one(current, df, Path(model_dir), Path(metric_dir), Path(plot_dir), Path(bi_dir))
        results[label] = metrics
    (Path(metric_dir) / "model_compare.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet generado por scripts/etl_attrition.py")
    ap.add_argument("--model", required=True, choices=[*MODELS, "all"])
    ap.add_argument("--outdir_models", default="/output/models")
    ap.add_argument("--outdir_metrics", default="/output/metrics")
    ap.add_argument("--outdir_plots", default="/output/plots")
    ap.add_argument("--outdir_bi", default="/output/bi")
    args = ap.parse_args()
    run(args.input, args.model, args.outdir_models, args.outdir_metrics, args.outdir_plots, args.outdir_bi)
