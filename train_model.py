"""Train numerical fraud models using creditcard.csv (Time, V1-V28, Amount, Class).
This dataset has no transaction-message text, so this model is separate from app.py's message-screening demo.
"""
from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
DATA_PATH = Path("creditcardlast.csv")
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Put creditcard.csv in the project root. Download link is in README.md.")
    print("Loading CSV...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")
    df = df.drop_duplicates().copy()
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df[df["Class"].isin([0, 1])].copy()
    df["Class"] = df["Class"].astype(int)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna(subset=["Class"])

    summary = {
        "rows_after_cleaning": int(len(df)),
        "columns": list(df.columns),
        "class_counts": {str(k): int(v) for k, v in df["Class"].value_counts().sort_index().items()},
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows_after_cleaning": int(df.duplicated().sum()),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df["Class"].value_counts().sort_index().rename_axis("Class").to_csv(OUT / "class_counts.csv")
    print(json.dumps(summary, indent=2))

    X, y = df.drop(columns=["Class"]), df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ]),
    }
    reports, best_name, best_model, best_ap = {}, None, None, -1
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, proba)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = None
        reports[name] = {
            "average_precision": float(ap),
            "roc_auc": float(auc) if auc is not None else None,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        print(f"Average precision: {ap:.4f}")
        if auc is not None:
            print(f"ROC AUC: {auc:.4f}")
        print(classification_report(y_test, pred, digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        if ap > best_ap:
            best_name, best_model, best_ap = name, model, ap

    joblib.dump(best_model, OUT / "numerical_fraud_model.joblib")
    (OUT / "model_metrics.json").write_text(json.dumps({"selected_model": best_name, "models": reports}, indent=2), encoding="utf-8")
    (OUT / "feature_names.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    print(f"\nSaved best numerical model: {best_name}")
    print(f"Artifacts: {OUT.resolve()}")
    print("Reminder: the numerical model needs all numerical dataset features; it does not accept a transaction message.")

if __name__ == "__main__":
    main()
