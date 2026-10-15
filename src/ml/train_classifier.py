"""
DWDM Project: Mumbai BMC Civic Complaint Classification
Trains and evaluates Supervised Learning models for Complaint Category Classification:
1. Logistic Regression (Baseline)
2. Random Forest Classifier
3. HistGradientBoosting Classifier (Gradient Boosting)

Target Leakage Prevention:
- Excludes `department_assigned` (1:1 with category)
- Excludes resolution-phase variables (resolution_days, complaint_status, contractor, work_quality, etc.)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Department routing lookup for Mumbai BMC
DEPT_ROUTING = {
    "Solid Waste / Garbage": "Solid Waste Management (SWM)",
    "Pothole / Road Damage": "Roads & Traffic Department",
    "Water Supply Disruption": "Hydraulic Engineer Department",
    "Water Leakage / Pipe Burst": "Hydraulic Engineer Department",
    "Drainage Overflow / Flooding": "Storm Water Drains (SWD)",
    "Street Light Failure": "Mechanical & Electrical Department",
    "Illegal Construction": "Building Proposal / Assessment",
    "Encroachment": "Encroachment Removal Cell",
    "Public Toilet Condition": "Sanitation & Public Health",
    "Tree Fallen / Dangerous Tree": "Garden & Tree Authority",
    "Noise / Air Pollution": "Environment & Pollution Control",
    "Stray Animal Menace": "Veterinary Public Health",
    "Health / Epidemic": "Public Health & Epidemiology",
}

NUMERIC_FEATURES = [
    "ward_slum_percentage",
    "is_monsoon_season",
    "has_photo_evidence",
    "has_gps_location",
    "media_attention",
    "politically_sensitive",
]

CATEGORICAL_FEATURES = [
    "ward_code",
    "zone",
    "ward_type",
    "population_density",
    "complaint_channel",
    "severity",
    "property_type",
    "complainant_type",
    "complaint_time_of_day",
]

TARGET = "complaint_category"


def load_data(sample_size=100000, random_state=42):
    train_path = os.path.join("dataset", "bmc_train.csv")
    print(f"Loading {sample_size} rows from {train_path}...")
    
    # Read chunk or sample
    df_sample = pd.read_csv(train_path, nrows=sample_size)
    
    # Fill any missing values safely
    for col in NUMERIC_FEATURES:
        df_sample[col] = pd.to_numeric(df_sample[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES:
        df_sample[col] = df_sample[col].astype(str).fillna("Unknown")
        
    X = df_sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df_sample[TARGET].astype(str)
    return X, y


def train_and_evaluate():
    X, y = load_data(sample_size=100000)
    classes = sorted(y.unique().tolist())
    print(f"Classes ({len(classes)}): {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    models = {
        "Logistic Regression Baseline": LogisticRegression(
            max_iter=500, random_state=42, solver="saga", tol=0.01
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, random_state=42
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        loss_val = float(log_loss(y_test, y_prob, labels=classes))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

        # Per-class metrics
        report_dict = classification_report(
            y_test, y_pred, labels=classes, output_dict=True, zero_division=0
        )
        per_class = []
        for c in classes:
            c_metrics = report_dict.get(c, {})
            per_class.append(
                {
                    "class_name": c,
                    "department": DEPT_ROUTING.get(c, "BMC Ward General"),
                    "support": int(c_metrics.get("support", 0)),
                    "precision": round(float(c_metrics.get("precision", 0.0)), 4),
                    "recall": round(float(c_metrics.get("recall", 0.0)), 4),
                    "f1_score": round(float(c_metrics.get("f1-score", 0.0)), 4),
                }
            )

        results[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "log_loss": round(loss_val, 4),
            "test_samples": len(y_test),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": [[round(float(v), 3) for v in row] for row in cm_norm],
            "per_class": per_class,
        }
        print(
            f"{name} -> Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}, Log-Loss: {loss_val:.4f}"
        )

        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_model_name = name
            best_pipeline = pipeline

    # Extract feature importance from Random Forest
    rf_pipeline = Pipeline(
        steps=[("prep", preprocessor), ("clf", models["Random Forest"])]
    )
    # Fit was already done
    ohe = best_pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    rf_clf = models["Random Forest"]
    importances = rf_clf.feature_importances_
    # Group one-hot importances back to high-level parent features
    parent_importance = {feat: 0.0 for feat in (NUMERIC_FEATURES + CATEGORICAL_FEATURES)}
    for name, imp in zip(all_feature_names, importances):
        for parent in CATEGORICAL_FEATURES:
            if name.startswith(parent + "_") or name == parent:
                parent_importance[parent] += float(imp)
                break
        else:
            if name in parent_importance:
                parent_importance[name] += float(imp)

    sorted_importance = [
        {"feature": k, "importance": round(v, 4)}
        for k, v in sorted(parent_importance.items(), key=lambda x: x[1], reverse=True)
    ]

    payload = {
        "classes": classes,
        "department_routing": DEPT_ROUTING,
        "best_model": best_model_name,
        "models": results,
        "feature_importance": sorted_importance,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }

    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "classifier_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("Saved evaluation metrics to models/classifier_metrics.json")

    # Save best pipeline model for live web inference
    joblib.dump(best_pipeline, os.path.join("models", "classifier_model.pkl"), compress=3)
    print(f"Saved best model pipeline ({best_model_name}) to models/classifier_model.pkl")


if __name__ == "__main__":
    train_and_evaluate()
