"""
Train a model to predict whether a person will seek PROFESSIONAL consultation
after self-diagnosing a mental health condition via social media content.

Target column: 'Professional Consultation Post-Self-Diagnosis' (Yes/No)
Dataset: data/survey.csv  (355 responses)

Run:
    python train_model.py
Outputs:
    model/model.joblib          -> trained sklearn Pipeline (preprocessing + classifier)
    model/metrics.json          -> evaluation metrics for the Streamlit app
    model/feature_importance.json
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / "data" / "survey.csv"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Professional Consultation Post-Self-Diagnosis"

# Top platforms / conditions used to build presence flags (multi-label columns)
TOP_PLATFORMS = ["TikTok", "Instagram", "YouTube", "Facebook", "Reddit", "X (Twitter)", "Snapchat", "WhatsApp"]
TOP_CONDITIONS = ["ADHD", "Anxiety", "Depression", "Autism", "Bipolar Disorder", "Digestive Issues"]

NUMERIC_FEATURES = [
    "Frequency of Medical Content Exposure (1-5)",
    "Perceived Danger of Self-Diagnosis (1-5)",
    "Content Simplifies Complex Issues (1-5)",
    "Num Platforms Used",
    "Num Conditions Self-Diagnosed",
]

CATEGORICAL_FEATURES = [
    "Age Group",
    "Daily Time Spent",
    "Search for Symptoms on Social Media Instead of Search Engines",
    "Algorithmic Recommendations of Diagnostic Content",
    "Action Taken When Seeing Matching Symptoms",
    "General Feeling After Watching Self-Diagnosis Content",
    "Verification of Content Creator Credibility",
    "Suspected Self-Condition Based Solely on Video",
    "Next Step After Suspecting Condition",
    "Purchased Medication/Supplements Based on SM Advice",
]

PLATFORM_FLAG_COLS = [f"platform_{p.replace(' ', '_').replace('(', '').replace(')', '')}" for p in TOP_PLATFORMS]
CONDITION_FLAG_COLS = [f"cond_{c.replace(' ', '_')}" for c in TOP_CONDITIONS]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Primary Social Media Platforms"] = df["Primary Social Media Platforms"].fillna("")
    df["Commonly Self-Diagnosed Conditions Observed"] = df[
        "Commonly Self-Diagnosed Conditions Observed"
    ].fillna("")

    for plat, col in zip(TOP_PLATFORMS, PLATFORM_FLAG_COLS):
        df[col] = df["Primary Social Media Platforms"].str.contains(plat, case=False, regex=False).astype(int)

    for cond, col in zip(TOP_CONDITIONS, CONDITION_FLAG_COLS):
        df[col] = (
            df["Commonly Self-Diagnosed Conditions Observed"].str.contains(cond, case=False, regex=False).astype(int)
        )

    df["Num Platforms Used"] = df["Primary Social Media Platforms"].apply(
        lambda x: len([p for p in x.split(",") if p.strip()])
    )
    df["Num Conditions Self-Diagnosed"] = df["Commonly Self-Diagnosed Conditions Observed"].apply(
        lambda x: len([c for c in x.split(",") if c.strip()])
    )

    df["Suspected Self-Condition Based Solely on Video"] = df[
        "Suspected Self-Condition Based Solely on Video"
    ].fillna("Not sure")

    return df


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    df = df[df[TARGET].isin(["Yes", "No"])].copy()
    y = (df[TARGET] == "Yes").astype(int)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PLATFORM_FLAG_COLS + CONDITION_FLAG_COLS
    X = df[feature_cols].copy()
    return X, y


def build_pipeline():
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES + PLATFORM_FLAG_COLS + CONDITION_FLAG_COLS),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    return pipeline


def main():
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Target balance -> Yes: {y.sum()} ({y.mean():.1%})  No: {(1 - y).sum()}")

    pipeline = build_pipeline()

    # Stratified 5-fold CV for robust evaluation on this small/imbalanced dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_cv = (y_proba_cv >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred_cv), 3),
        "precision": round(precision_score(y, y_pred_cv, zero_division=0), 3),
        "recall": round(recall_score(y, y_pred_cv, zero_division=0), 3),
        "f1": round(f1_score(y, y_pred_cv, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y, y_proba_cv), 3),
        "confusion_matrix": confusion_matrix(y, y_pred_cv).tolist(),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "classification_report": classification_report(y, y_pred_cv, target_names=["No", "Yes"], output_dict=True),
    }
    print(json.dumps(metrics, indent=2))

    # Fit final model on ALL data for deployment
    pipeline.fit(X, y)

    # Feature importance (map back to original feature names via preprocessor)
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    num_names = NUMERIC_FEATURES + PLATFORM_FLAG_COLS + CONDITION_FLAG_COLS
    all_names = list(num_names) + list(cat_names)
    importances = pipeline.named_steps["classifier"].feature_importances_
    fi = sorted(zip(all_names, importances), key=lambda t: t[1], reverse=True)[:15]
    feature_importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in fi]

    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    joblib.dump(
        {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "platform_flags": PLATFORM_FLAG_COLS,
            "condition_flags": CONDITION_FLAG_COLS,
            "top_platforms": TOP_PLATFORMS,
            "top_conditions": TOP_CONDITIONS,
        },
        MODEL_DIR / "feature_schema.joblib",
    )
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    print("\nSaved model + metrics to", MODEL_DIR)


if __name__ == "__main__":
    main()
