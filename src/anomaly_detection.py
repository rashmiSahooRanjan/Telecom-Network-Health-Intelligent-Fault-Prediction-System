"""
Telecom Network Health & Intelligent Fault Prediction System
Anomaly Detection Module: Scikit-learn Isolation Forest unsupervised anomaly detector.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@st.cache_resource(show_spinner=False)
def train_isolation_forest(
    X_train: np.ndarray, 
    contamination: float = 0.05, 
    random_state: int = 42
) -> IsolationForest:
    """
    Train an Isolation Forest model on scaled numeric KPI features.
    Cached via st.cache_resource for maximum performance.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def detect_anomalies(
    df: pd.DataFrame, 
    kpi_features: List[str], 
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end unsupervised anomaly detection pipeline:
    1. Extract and impute numeric KPI columns
    2. Normalize using StandardScaler
    3. Train / predict with Isolation Forest
    4. Compute anomaly scores (lower = more abnormal)
    5. Output enriched DataFrame with 'is_anomaly' and 'anomaly_score'
    
    Returns:
        (df_with_anomaly_tags, summary_metrics)
    """
    if df.empty or not kpi_features:
        return df.copy(), {"total": 0, "anomalies": 0, "rate": 0.0}

    # Extract feature matrix
    X_df = df[kpi_features].copy()
    for col in kpi_features:
        if X_df[col].isnull().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())
            
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    # Train model
    model = train_isolation_forest(X_scaled, contamination=contamination)
    
    # -1 indicates anomaly, 1 indicates normal in Scikit-Learn IsolationForest
    raw_preds = model.predict(X_scaled)
    # decision_function: negative scores represent anomalies
    scores = model.decision_function(X_scaled)

    out = df.copy()
    out["is_anomaly"] = [1 if p == -1 else 0 for p in raw_preds]
    out["anomaly_label"] = ["Anomaly" if p == -1 else "Normal" for p in raw_preds]
    out["anomaly_score"] = np.round(scores, 4)

    total_records = len(out)
    anomaly_count = int((out["is_anomaly"] == 1).sum())
    normal_count = total_records - anomaly_count
    anomaly_rate = (anomaly_count / total_records * 100.0) if total_records > 0 else 0.0

    summary = {
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_rate_pct": round(anomaly_rate, 2),
        "contamination": contamination,
        "features_used": kpi_features,
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_score": float(np.mean(scores))
    }

    return out, summary


def get_anomaly_kpi_comparison(
    df_results: pd.DataFrame, 
    kpi_features: List[str]
) -> pd.DataFrame:
    """
    Compare average KPI values between Normal and Anomalous samples.
    Computes percentage delta to identify the key KPIs driving abnormalities.
    """
    if "is_anomaly" not in df_results.columns or df_results.empty:
        return pd.DataFrame()

    normal_df = df_results[df_results["is_anomaly"] == 0]
    anomaly_df = df_results[df_results["is_anomaly"] == 1]

    if normal_df.empty or anomaly_df.empty:
        return pd.DataFrame()

    comparison_rows = []
    for kpi in kpi_features:
        if kpi in df_results.columns:
            norm_mean = float(normal_df[kpi].mean())
            anom_mean = float(anomaly_df[kpi].mean())
            diff = anom_mean - norm_mean
            pct_change = (diff / abs(norm_mean) * 100.0) if norm_mean != 0 else np.nan
            
            comparison_rows.append({
                "KPI": kpi,
                "Normal Mean": round(norm_mean, 3),
                "Anomaly Mean": round(anom_mean, 3),
                "Absolute Diff": round(diff, 3),
                "% Change": round(pct_change, 1) if not pd.isna(pct_change) else 0.0
            })

    return pd.DataFrame(comparison_rows)
