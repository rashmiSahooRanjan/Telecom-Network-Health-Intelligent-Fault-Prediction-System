"""
Telecom Network Health & Intelligent Fault Prediction System
Prediction Engine: Real-time inference, risk level assessment, and contributing KPI diagnostics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from src.network_health import calculate_single_health_score
from src.recommendations import generate_recommendations


def predict_single_sample(
    input_kpis: Dict[str, float],
    model: Any,
    scaler: Any,
    feature_cols: List[str],
    label_encoder: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Perform real-time fault/condition inference on user-supplied KPI telemetry.
    
    Returns structured prediction diagnostic:
      - Predicted Condition / Class
      - Confidence / Probability
      - Risk Level (Low, Medium, High, Critical)
      - Network Health Score (0-100)
      - Penalties and Contributing KPIs
      - Recommended Next Troubleshooting Actions
    """
    # Build vector in correct feature order
    feature_vals = []
    for col in feature_cols:
        val = input_kpis.get(col, 0.0)
        feature_vals.append(val)
        
    X_raw = np.array([feature_vals])
    X_scaled = scaler.transform(X_raw) if scaler else X_raw

    # Model prediction
    raw_pred = model.predict(X_scaled)[0]
    
    # Class probability if supported
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probas))
        except Exception:
            confidence = None

    if label_encoder and hasattr(label_encoder, "inverse_transform"):
        try:
            predicted_label = str(label_encoder.inverse_transform([raw_pred])[0])
        except Exception:
            predicted_label = str(raw_pred)
    else:
        predicted_label = str(raw_pred)

    # Compute transparent health score and rule-based assessment
    series_input = pd.Series(input_kpis)
    health_score, health_status, penalties, reasons = calculate_single_health_score(series_input)

    # Determine risk level based on health score and model prediction
    is_anomaly_predicted = (
        predicted_label.lower() in ["yes", "1", "anomaly", "true"] or 
        (predicted_label.lower() not in ["no", "0", "normal", "false", "healthy"])
    )

    if health_score < 50 or (is_anomaly_predicted and health_score < 70):
        risk_level = "Critical"
        risk_color = "#ef4444"
    elif health_score < 70 or is_anomaly_predicted:
        risk_level = "High"
        risk_color = "#f97316"
    elif health_score < 90:
        risk_level = "Medium"
        risk_color = "#eab308"
    else:
        risk_level = "Low"
        risk_color = "#22c55e"

    # Generate troubleshooting recommendations
    recs = generate_recommendations(series_input)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "health_score": round(health_score, 1),
        "health_status": health_status,
        "penalties": penalties,
        "contributing_reasons": reasons,
        "recommendations": recs,
        "input_features": input_kpis
    }
