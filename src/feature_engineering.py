"""
Telecom Network Health & Intelligent Fault Prediction System
Feature Engineering Module: Domain-specific telecom metric derivation and composite performance indices.
"""

import pandas as pd
import numpy as np


def engineer_telecom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive domain-specific 5G radio and transport network indices:
    - Combined DL/UL Throughput (in KB and Mbps)
    - Block Error Rate / Packet Loss percentage
    - Composite Radio Quality Score (from RSRP and SNR)
    - Peak PRB Congestion index
    - Modulation efficiency score
    """
    if df.empty:
        return df.copy()
        
    out = df.copy()
    
    # 1. Total Throughput
    has_tx = "TX_Bytes" in out.columns
    has_rx = "RX_Bytes" in out.columns
    if has_tx and has_rx:
        out["Throughput_Total_KB"] = (out["TX_Bytes"] + out["RX_Bytes"]) / 1024.0
        
        # Calculate Mbps if sampling rate exists (default 10s window in TelecomTS)
        sampling_sec = out["sampling_rate"] if "sampling_rate" in out.columns else 10.0
        out["Throughput_Mbps"] = ((out["TX_Bytes"] + out["RX_Bytes"]) * 8.0) / (sampling_sec * 1_000_000.0)
    elif has_tx:
        out["Throughput_Total_KB"] = out["TX_Bytes"] / 1024.0
    elif has_rx:
        out["Throughput_Total_KB"] = out["RX_Bytes"] / 1024.0

    # 2. Packet Loss & Error Rate
    if "DL_BLER" in out.columns:
        out["DL_Packet_Loss_Pct"] = (out["DL_BLER"] * 100.0).clip(0.0, 100.0)
    if "UL_BLER" in out.columns:
        out["UL_Packet_Loss_Pct"] = (out["UL_BLER"] * 100.0).clip(0.0, 100.0)
    if "DL_BLER" in out.columns and "UL_BLER" in out.columns:
        out["Max_BLER_Pct"] = np.maximum(out["DL_BLER"], out["UL_BLER"]) * 100.0

    # 3. Peak PRB Congestion Index (0 to 100%)
    if "PRB_Utilization_DL" in out.columns and "PRB_Utilization_UL" in out.columns:
        out["Peak_PRB_Utilization"] = np.maximum(out["PRB_Utilization_DL"], out["PRB_Utilization_UL"])
    elif "PRB_Utilization_DL" in out.columns:
        out["Peak_PRB_Utilization"] = out["PRB_Utilization_DL"]

    # 4. Composite Radio Quality Index (0 - 100)
    # RSRP: typical usable range is -125 dBm (edge/dead) to -65 dBm (excellent)
    # SNR: typical range 0 dB (very poor) to 30 dB (excellent)
    if "RSRP" in out.columns and "UL_SNR" in out.columns:
        norm_rsrp = ((out["RSRP"] - (-125.0)) / ((-65.0) - (-125.0))).clip(0.0, 1.0) * 100.0
        norm_snr = (out["UL_SNR"] / 30.0).clip(0.0, 1.0) * 100.0
        out["Radio_Quality_Index"] = (norm_rsrp * 0.5) + (norm_snr * 0.5)
    elif "RSRP" in out.columns:
        out["Radio_Quality_Index"] = (((out["RSRP"] - (-125.0)) / 60.0).clip(0.0, 1.0) * 100.0)

    # 5. Modulation & Coding Scheme (MCS) Average
    if "DL_MCS" in out.columns and "UL_MCS" in out.columns:
        out["Mean_MCS"] = (out["DL_MCS"] + out["UL_MCS"]) / 2.0
    elif "DL_MCS" in out.columns:
        out["Mean_MCS"] = out["DL_MCS"]

    return out
