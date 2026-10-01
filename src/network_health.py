"""
Telecom Network Health & Intelligent Fault Prediction System
Network Health Scoring Engine: Configurable, rule-transparent KPI health evaluator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List


# Default benchmark thresholds (Configurable in UI)
DEFAULT_HEALTH_THRESHOLDS = {
    "bler_warning": 0.05,       # 5% BLER warning
    "bler_critical": 0.15,      # 15% BLER critical
    "rsrp_warning": -95.0,      # -95 dBm warning
    "rsrp_critical": -110.0,    # -110 dBm critical
    "snr_warning": 12.0,        # 12 dB warning
    "snr_critical": 5.0,        # 5 dB critical
    "prb_util_warning": 75.0,   # 75% utilization warning
    "prb_util_critical": 90.0,  # 90% utilization critical
    "buffer_warning": 20000.0,  # 20 KB buffer backlog warning
    "buffer_critical": 50000.0, # 50 KB buffer backlog critical
    "mcs_warning": 10.0,        # MCS index 10 warning
    "mcs_critical": 5.0         # MCS index 5 critical
}


def calculate_single_health_score(
    row: pd.Series, 
    thresholds: Dict[str, float] = DEFAULT_HEALTH_THRESHOLDS
) -> Tuple[float, str, Dict[str, float], List[str]]:
    """
    Calculate transparent Network Health Score (0 - 100) for a single record.
    
    Formula starts at 100. Penalties are deducted transparently based only
    on existing KPI evidence:
      - BLER (Error/Loss rate): up to -30 pts
      - Signal Quality (RSRP/SNR): up to -25 pts
      - Network Congestion (PRB Util): up to -20 pts
      - Buffer backlog (Queue Delay Proxy): up to -15 pts
      - Modulation/Efficiency (MCS): up to -10 pts
      
    Returns:
        (health_score, status_category, penalty_breakdown, contributing_reasons)
    """
    score = 100.0
    penalties = {}
    reasons = []

    # 1. Error Rate / Packet Loss (DL_BLER or UL_BLER)
    bler = None
    if "DL_BLER" in row and not pd.isna(row["DL_BLER"]):
        bler = float(row["DL_BLER"])
    elif "Packet_Loss_Rate_Pct" in row and not pd.isna(row["Packet_Loss_Rate_Pct"]):
        bler = float(row["Packet_Loss_Rate_Pct"]) / 100.0

    if bler is not None:
        if bler >= thresholds.get("bler_critical", 0.15):
            penalties["Block Error Rate"] = 30.0
            reasons.append(f"Severe Block Error Rate ({bler*100:.1f}%) exceeds critical threshold.")
        elif bler >= thresholds.get("bler_warning", 0.05):
            penalties["Block Error Rate"] = 15.0
            reasons.append(f"Elevated Block Error Rate ({bler*100:.1f}%) exceeds warning threshold.")

    # 2. Radio Signal Power (RSRP)
    if "RSRP" in row and not pd.isna(row["RSRP"]):
        rsrp = float(row["RSRP"])
        if rsrp <= thresholds.get("rsrp_critical", -110.0):
            penalties["RSRP (Signal Power)"] = 15.0
            reasons.append(f"Critical Signal Degradation: RSRP at {rsrp:.1f} dBm.")
        elif rsrp <= thresholds.get("rsrp_warning", -95.0):
            penalties["RSRP (Signal Power)"] = 8.0
            reasons.append(f"Marginal Signal Power: RSRP at {rsrp:.1f} dBm.")

    # 3. Radio Signal Quality (SNR)
    if "UL_SNR" in row and not pd.isna(row["UL_SNR"]):
        snr = float(row["UL_SNR"])
        if snr <= thresholds.get("snr_critical", 5.0):
            penalties["SNR (Signal Quality)"] = 10.0
            reasons.append(f"High Interference / Low SNR: {snr:.1f} dB.")
        elif snr <= thresholds.get("snr_warning", 12.0):
            penalties["SNR (Signal Quality)"] = 5.0
            reasons.append(f"Suboptimal SNR: {snr:.1f} dB.")

    # 4. PRB Resource Utilization (Congestion)
    prb_util = None
    if "PRB_Utilization_DL" in row and not pd.isna(row["PRB_Utilization_DL"]):
        prb_util = float(row["PRB_Utilization_DL"])
    elif "Peak_PRB_Utilization" in row and not pd.isna(row["Peak_PRB_Utilization"]):
        prb_util = float(row["Peak_PRB_Utilization"])

    if prb_util is not None:
        if prb_util >= thresholds.get("prb_util_critical", 90.0):
            penalties["PRB Congestion"] = 20.0
            reasons.append(f"Critical Radio Congestion: PRB Utilization at {prb_util:.1f}%.")
        elif prb_util >= thresholds.get("prb_util_warning", 75.0):
            penalties["PRB Congestion"] = 10.0
            reasons.append(f"High Radio Load: PRB Utilization at {prb_util:.1f}%.")

    # 5. Buffer Backlog (Queueing / Latency proxy)
    if "Estimated_UL_Buffer" in row and not pd.isna(row["Estimated_UL_Buffer"]):
        buffer_val = float(row["Estimated_UL_Buffer"])
        if buffer_val >= thresholds.get("buffer_critical", 50000.0):
            penalties["Buffer Backlog"] = 15.0
            reasons.append(f"Excessive buffer backlog queue ({buffer_val:,.0f} B).")
        elif buffer_val >= thresholds.get("buffer_warning", 20000.0):
            penalties["Buffer Backlog"] = 7.0
            reasons.append(f"Elevated buffer backlog ({buffer_val:,.0f} B).")

    # 6. Modulation & Coding Scheme (MCS)
    if "DL_MCS" in row and not pd.isna(row["DL_MCS"]):
        mcs = float(row["DL_MCS"])
        if mcs <= thresholds.get("mcs_critical", 5.0):
            penalties["Modulation (MCS)"] = 10.0
            reasons.append(f"Severe MCS degradation to index {mcs:.1f} (fallback mode).")
        elif mcs <= thresholds.get("mcs_warning", 10.0):
            penalties["Modulation (MCS)"] = 5.0
            reasons.append(f"Sub-optimal modulation scheme (MCS {mcs:.1f}).")

    total_penalty = sum(penalties.values())
    final_score = max(0.0, min(100.0, score - total_penalty))

    if final_score >= 90.0:
        status = "Healthy"
    elif final_score >= 70.0:
        status = "Good"
    elif final_score >= 50.0:
        status = "Warning"
    else:
        status = "Critical"

    return final_score, status, penalties, reasons


def compute_dataset_health(
    df: pd.DataFrame, 
    thresholds: Dict[str, float] = DEFAULT_HEALTH_THRESHOLDS
) -> pd.DataFrame:
    """
    Compute Network Health Score and category for all records in a DataFrame.
    Appends 'Health_Score' and 'Health_Status' columns.
    """
    if df.empty:
        return df.copy()

    scores = []
    statuses = []
    
    for _, row in df.iterrows():
        sc, st, _, _ = calculate_single_health_score(row, thresholds)
        scores.append(round(sc, 1))
        statuses.append(st)

    out = df.copy()
    out["Health_Score"] = scores
    out["Health_Status"] = statuses
    return out
