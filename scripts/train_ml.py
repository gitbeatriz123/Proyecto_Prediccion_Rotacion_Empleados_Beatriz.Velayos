import argparse, json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

EXCLUDE = {
    "EmployeeNumber", "EmployeeCount", "StandardHours", "Over18",
    "Attrition", "attrition_label",
}

def evaluate_probs(y, probs):
    auc = roc_auc_score(y, probs)
    ap  = average_precision_score(y, probs)
    prec, rec, th = precision_recall_curve(y, probs)
    f1s = (2*prec*rec) / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    thr_opt = float(th[best_idx-1]) if best_idx > 0 and best_idx-1 < len(th) else 0.5
    f1_opt  = float(f1s[best_idx])
    return float(auc), float(ap), thr_opt, f1_opt

def plot_and_save_curves(y, probs, name, out_plots: Path):
    out_plots.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC")
    plt.tight_layout(); plt.savefig(out_plots / f"{name.lower()}_roc.png", dpi=160); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} PR")
    plt.tight_layout(); plt.savefig(out_plots / f"{name.lower()}_pr.png", dpi=160); plt.close()

def run(path_in: str, which: str,
        out_models="/output/models", out_metrics="/output/metrics", out_plots="/output/plots"):
    out_models = Path(out_models); out_metrics = Path(out_metrics); out_plots = Path(out_plots)
    out_models.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(path_in)
    y  = df["attrition_label"].astype(int).values
    X  = df.drop(columns=[c for c in df.columns if c in EXCLUDE])

    num_cols = X.select_dtypes(include=[np.number, "Int64", "Float64", "boolean", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [("num", StandardScaler(with_mean=False), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop"
    )

    if which.lower() == "logreg":
        clf  = LogisticRegression(solver="saga", penalty="l2", max_iter=2000,
                                  class_weight="balanced", random_state=42)
        name = "LogReg"
    elif which.lower() == "rf":
        clf  = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      random_state=42)
        name = "RF"
    else:
        raise ValueError("modelo no soportado (usa: logreg | rf)")

    pipe = Pipeline([("prep", pre), ("clf", clf)])

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    auc, ap, thr_opt, f1_opt = evaluate_probs(y, probs)
    plot_and_save_curves(y, probs, name, out_plots)

    pipe.fit(X, y)
    joblib.dump(pipe, out_models / f"{name.lower()}_pipeline.pkl")

    cmp_path = out_metrics / "model_compare.json"
    compare = {}
    if cmp_path.exists():
        try:
            compare = json.loads(cmp_path.read_text())
        except Exception:
            compare = {}
    compare[name] = {"roc_auc": auc, "pr_auc": ap, "f1_opt": f1_opt, "thr_opt": thr_opt}
    cmp_path.write_text(json.dumps(compare, indent=2))

    (out_metrics / f"threshold_tuning_{name.lower()}.json").write_text(
        json.dumps({"best_threshold": thr_opt, "f1": f1_opt}, indent=2)
    )

    print(f"{name} -> AUC:{auc:.3f} | PR-AUC:{ap:.3f} | F1*:{f1_opt:.3f} | thr*:{thr_opt:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet procesado")
    ap.add_argument("--model", required=True, choices=["logreg", "rf"])
    ap.add_argument("--outdir_models", default="/output/models")
    ap.add_argument("--outdir_metrics", default="/output/metrics")
    ap.add_argument("--outdir_plots",  default="/output/plots")
    args = ap.parse_args()
    run(args.input, args.model, args.outdir_models, args.outdir_metrics, args.outdir_plots)
