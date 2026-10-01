"""
Telecom Network Health & Intelligent Fault Prediction System
Supervised Machine Learning Module: Multi-model training, evaluation, comparison, and feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder


@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    is_binary: bool = True
) -> Dict[str, Any]:
    """
    Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting.
    Cached via st.cache_resource.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}
    for name, clf in models.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Compute evaluation metrics
            avg_mode = "binary" if is_binary else "weighted"
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg_mode, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg_mode, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg_mode, zero_division=0)
            
            roc = None
            if is_binary and hasattr(clf, "predict_proba"):
                try:
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    roc = roc_auc_score(y_test, y_prob)
                except Exception:
                    roc = None

            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "model": clf,
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc), 4) if roc is not None else "N/A",
                "confusion_matrix": cm,
                "predictions": y_pred
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def run_supervised_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    End-to-end supervised ML training pipeline:
    1. Filter valid rows
    2. Encode target and scale features
    3. Stratified split
    4. Train 3 industry-standard classifiers
    5. Rank and select best model based on F1-score
    6. Extract feature importances
    """
    if df.empty or target_col not in df.columns or not feature_cols:
        raise ValueError("Invalid DataFrame or missing features/target column.")

    # Prepare data
    clean_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    for col in feature_cols:
        if clean_df[col].isnull().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    X = clean_df[feature_cols].values
    y_raw = clean_df[target_col].astype(str).values

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    classes = list(label_enc.classes_)
    is_binary = len(classes) == 2

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified Train/Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Fallback without stratification if any class has only 1 sample
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

    # Train and evaluate models
    model_results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, is_binary=is_binary
    )

    # Comparison summary table
    comparison_rows = []
    best_model_name = None
    best_f1 = -1.0

    for name, res in model_results.items():
        if "error" not in res:
            comparison_rows.append({
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1_score"],
                "ROC-AUC": res["roc_auc"]
            })
            if res["f1_score"] > best_f1:
                best_f1 = res["f1_score"]
                best_model_name = name

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    # Feature importances from best tree model
    feature_importances = {}
    best_clf = model_results.get(best_model_name, {}).get("model") if best_model_name else None
    if best_clf and hasattr(best_clf, "feature_importances_"):
        fi = best_clf.feature_importances_
        feature_importances = dict(zip(feature_cols, [round(float(v), 4) for v in fi]))
        # Sort descending
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    return {
        "comparison_table": comparison_df,
        "best_model_name": best_model_name,
        "best_f1_score": best_f1,
        "model_results": model_results,
        "classes": classes,
        "is_binary": is_binary,
        "feature_importances": feature_importances,
        "scaler": scaler,
        "label_encoder": label_enc,
        "test_labels": y_test,
        "feature_cols": feature_cols
    }
