"""
Telecom Network Health & Intelligent Fault Prediction System
Preprocessing Module: Data validation, missing value imputation, type casting, and KPI column detection.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Recognized core 5G telecom KPI features in TelecomTS and standard telecom telemetry
TELECOM_KPI_CANDIDATES = [
    "RSRP", "UL_SNR", "DL_BLER", "UL_BLER", "DL_MCS", "UL_MCS",
    "TX_Bytes", "RX_Bytes", "PRB_Utilization_DL", "PRB_Utilization_UL",
    "Estimated_UL_Buffer", "UL_NPRB", "PRBs_DL_Current", "PRBs_UL_Current",
    "UL_NumberOfPackets", "DL_NumberOfPackets", "Throughput_Total_KB",
    "Packet_Loss_Rate_Pct", "Latency_ms"
]


def detect_kpi_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically identify numeric KPI columns present in a DataFrame.
    Prioritizes known telecom metrics, followed by any valid floating/integer telemetry.
    """
    if df.empty:
        return []
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out internal IDs and metadata
    excluded = {"record_id", "id", "sampling_rate", "step", "unnamed: 0"}
    candidates = [c for c in numeric_cols if c.lower() not in excluded and not c.endswith("_std")]
    
    # Prioritize known telecom KPIs
    prioritized = [c for c in TELECOM_KPI_CANDIDATES if c in candidates]
    remaining = [c for c in candidates if c not in prioritized]
    
    return prioritized + remaining


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning:
    - Handle missing values (numeric median, categorical 'Unknown')
    - Strip whitespace from categorical strings
    - Cast timestamp if valid
    - Remove exact duplicate rows
    """
    if df.empty:
        return df.copy()
        
    cleaned = df.copy()
    
    # Parse timestamp safely
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
        
    # Remove exact duplicates
    cleaned = cleaned.drop_duplicates()
    
    # Handle numeric columns
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if cleaned[col].isnull().any():
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            
    # Handle categorical / object columns
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
        cleaned[col] = cleaned[col].fillna("Unknown")
        
    return cleaned


def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    """
    Validate an uploaded CSV file from user.
    Checks for minimum rows and identifies numeric columns suitable for KPI analysis.
    
    Returns:
        (is_valid, message, detected_kpi_columns)
    """
    if df is None or df.empty:
        return False, "The uploaded CSV file is completely empty.", []
        
    if len(df) < 5:
        return False, "The uploaded CSV contains fewer than 5 records. Please provide a larger sample.", []
        
    detected_kpis = detect_kpi_columns(df)
    if not detected_kpis:
        return False, "No numeric telemetry or KPI columns were detected in the uploaded file.", []
        
    return True, f"Valid CSV. Detected {len(detected_kpis)} numeric KPI column(s).", detected_kpis


def preprocess_for_ml(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler, Dict[str, Any]]:
    """
    Standardize feature matrix and encode optional target column for ML training.
    
    Returns:
        (X_scaled, y_encoded, scaler, metadata)
    """
    if df.empty or not feature_cols:
        raise ValueError("DataFrame is empty or no feature columns were provided.")
        
    # Extract features and fill remaining NaNs with column medians
    X_df = df[feature_cols].copy()
    for c in feature_cols:
        if X_df[c].isnull().any():
            X_df[c] = X_df[c].fillna(X_df[c].median())
            
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    
    y = None
    target_encoder = None
    target_classes = []
    
    if target_col and target_col in df.columns:
        y_raw = df[target_col].copy()
        if y_raw.dtype == "object" or isinstance(y_raw.iloc[0], str):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_raw.astype(str))
            target_classes = list(target_encoder.classes_)
        else:
            y = y_raw.values
            
    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_encoder": target_encoder,
        "classes": target_classes
    }
    
    return X_scaled, y, scaler, metadata
