"""Entrena y evalúa modelos con validación cruzada anidada y umbral seleccionado sin contaminar el fold externo."""
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
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDE = {"EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "Attrition", "attrition_label", "income_yearly", "overtime_flag"}
MODELS = ("logreg", "rf", "mlp")

def select_threshold(y, probs):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    if len(thresholds) == 0:
        return 0.5, float(f1_score(y, (probs >= 0.5).astype(int), zero_division=0))
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])

def make_pipeline(model_name, numeric, categorical):
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="drop")
    if model_name == "logreg":
        clf, label = LogisticRegression(solver="saga", penalty="l2", max_iter=3000, class_weight="balanced", random_state=42), "LogReg"
    elif model_name == "rf":
        clf, label = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1), "RF"
    else:
        clf, label = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=150, early_stopping=True, random_state=42), "MLP"
    return Pipeline([("prep", pre), ("clf", clf)]), label

def save_curves(y, probs, name, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC"); plt.tight_layout()
    plt.savefig(plots_dir / f"{name.lower()}_roc.png", dpi=160); plt.close()
    precision, recall, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(5, 4)); plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} Precision–Recall"); plt.tight_layout()
    plt.savefig(plots_dir / f"{name.lower()}_pr.png", dpi=160); plt.close()

def nested_oof(pipe, X, y):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.empty(len(y), dtype=float)
    thresholds = []
    for train_idx, test_idx in outer.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_probs = cross_val_predict(pipe, X_train, y_train, cv=inner, method="predict_proba", n_jobs=1)[:, 1]
        threshold, _ = select_threshold(y_train, inner_probs)
        thresholds.append(threshold)
        pipe.fit(X_train, y_train)
        probs[test_idx] = pipe.predict_proba(X_test)[:, 1]
    return probs, np.asarray(thresholds)

def evaluate_probs(y, probs, threshold_reference):
    pred = (probs >= threshold_reference).astype(int)
    pred05 = (probs >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "f1_opt": float(f1_score(y, pred, zero_division=0)),
        "thr_opt": float(threshold_reference),
        "accuracy_at_0_5": float(accuracy_score(y, pred05)),
        "f1_at_0_5": float(f1_score(y, pred05, zero_division=0)),
        "protocol": "nested_5fold_outer_3fold_inner_threshold_selection",
    }

def train_one(model_name, df, model_dir, metric_dir, plot_dir, bi_dir):
    y = df["attrition_label"].astype(int).to_numpy()
    ids = df["EmployeeNumber"].to_numpy() if "EmployeeNumber" in df else np.arange(len(df))
    X = df.drop(columns=[c for c in df.columns if c in EXCLUDE])
    numeric = X.select_dtypes(include=[np.number, "Int64", "Float64", "boolean", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pipe, label = make_pipeline(model_name, numeric, categorical)
    probs, thresholds = nested_oof(pipe, X, y)
    threshold_reference = float(np.median(thresholds))
    metrics = evaluate_probs(y, probs, threshold_reference)
    metrics["threshold_reference"] = threshold_reference
    metrics["threshold_selection"] = [float(x) for x in thresholds]
    save_curves(y, probs, label, plot_dir)
    pipe.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True); metric_dir.mkdir(parents=True, exist_ok=True); bi_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold_reference}, model_dir / f"{model_name}_pipeline.pkl")
    (metric_dir / f"{model_name}_oof_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if model_name == "logreg":
        out = pd.DataFrame({"employee_number": ids, "attrition_label": y, "proba": probs, "pred": (probs >= threshold_reference).astype(int)})
        for source, target in (("Department","department"),("JobRole","jobrole"),("Gender","gender"),("Age","age"),("MonthlyIncome","monthlyincome"),("YearsAtCompany","yearsatcompany"),("OverTime","overtime")):
            if source in df.columns: out[target] = df[source].to_numpy()
        out.to_csv(bi_dir / "predictions_logreg.csv", index=False)
        names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pipe.named_steps["clf"].coef_.ravel()
        pd.DataFrame({"feature":[n.split("__",1)[-1] for n in names],"effect":coefs}).sort_values("effect", key=lambda s:s.abs(), ascending=False).to_csv(bi_dir / "feature_effects.csv", index=False)
    if model_name == "rf":
        names = pipe.named_steps["prep"].get_feature_names_out()
        pd.DataFrame({"feature":[n.split("__",1)[-1] for n in names],"importance":pipe.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False).to_csv(metric_dir / "feature_importance_rf.csv", index=False)
    print(f"{label}: ROC-AUC={metrics['roc_auc']:.3f}; PR-AUC={metrics['pr_auc']:.3f}; F1 evaluado={metrics['f1_opt']:.3f}; umbral referencia={threshold_reference:.3f}")
    return label, metrics

def run(input_path, model_name, model_dir, metric_dir, plot_dir, bi_dir):
    df = pd.read_parquet(input_path)
    if "attrition_label" not in df: raise ValueError("Falta attrition_label en el Parquet.")
    results = {}
    for current in MODELS if model_name == "all" else (model_name,):
        label, metrics = train_one(current, df, Path(model_dir), Path(metric_dir), Path(plot_dir), Path(bi_dir))
        results[label] = metrics
    Path(metric_dir).mkdir(parents=True, exist_ok=True)
    (Path(metric_dir) / "model_compare.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--model", required=True, choices=[*MODELS,"all"])
    ap.add_argument("--outdir_models", default="/output/models"); ap.add_argument("--outdir_metrics", default="/output/metrics")
    ap.add_argument("--outdir_plots", default="/output/plots"); ap.add_argument("--outdir_bi", default="/output/bi")
    args = ap.parse_args()
    run(args.input, args.model, args.outdir_models, args.outdir_metrics, args.outdir_plots, args.outdir_bi)
