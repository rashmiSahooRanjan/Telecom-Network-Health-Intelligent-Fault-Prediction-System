"""
Telecom Network Health & Intelligent Fault Prediction System
Data Loader Module: Hugging Face TelecomTS loader, schema inspector, and KPI DataFrame extractor.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


SAMPLE_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "telecomts_sample.csv")


@st.cache_data(show_spinner=False)
def load_telecom_dataset_hf(sample_limit: Optional[int] = None) -> Tuple[Optional[Any], str]:
    """
    Load TelecomTS directly from Hugging Face dataset repository.
    Uses pattern data_files={'full': '**/chunked.jsonl'}.
    
    Returns:
        (dataset_dict, status_message)
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("AliMaatouk/TelecomTS", data_files={"full": "**/chunked.jsonl"})
        total_rows = len(ds["full"])
        return ds, f"Successfully loaded {total_rows:,} records from Hugging Face (AliMaatouk/TelecomTS)."
    except Exception as e:
        err_msg = (
            "Unable to load TelecomTS from Hugging Face. "
            "Please check your internet connection or Hugging Face availability. "
            f"(Details: {str(e)[:150]})"
        )
        return None, err_msg


@st.cache_data(show_spinner=False)
def load_sample_dataset() -> pd.DataFrame:
    """Load local pre-extracted TelecomTS representative dataset with error handling."""
    if os.path.exists(SAMPLE_CSV_PATH):
        try:
            df = pd.read_csv(SAMPLE_CSV_PATH)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def inspect_dataset(ds) -> Dict[str, Any]:
    """
    Inspect schema and structure of raw Hugging Face TelecomTS dataset.
    Returns dictionary with features, sample row preview, and split statistics.
    """
    if ds is None:
        return {"status": "Not loaded"}
        
    try:
        full_split = ds["full"]
        num_rows = len(full_split)
        sample = full_split[0]
        
        kpis_available = []
        if "statistics" in sample and isinstance(sample["statistics"], dict):
            kpis_available = list(sample["statistics"].keys())
        elif "KPIs" in sample and isinstance(sample["KPIs"], dict):
            kpis_available = list(sample["KPIs"].keys())

        return {
            "num_rows": num_rows,
            "splits": list(ds.keys()),
            "features": list(sample.keys()),
            "kpis": kpis_available,
            "sample_keys": list(sample.keys()),
            "labels_schema": list(sample.get("labels", {}).keys()) if isinstance(sample.get("labels"), dict) else [],
            "anomalies_schema": list(sample.get("anomalies", {}).keys()) if isinstance(sample.get("anomalies"), dict) else [],
            "status": "Loaded"
        }
    except Exception as e:
        return {"status": "Error", "error": str(e)}


def extract_kpi_dataframe(ds_or_split, max_records: Optional[int] = 2000) -> pd.DataFrame:
    """
    Convert structured/nested TelecomTS JSON records into a clean, flat Pandas DataFrame.
    Safely extracts all numeric KPIs, standard deviations, network conditions,
    anomaly tags, and troubleshooting summaries.
    """
    if ds_or_split is None:
        return pd.DataFrame()
        
    split = ds_or_split["full"] if hasattr(ds_or_split, "__getitem__") and "full" in ds_or_split else ds_or_split
    total_len = len(split)
    
    limit = min(total_len, max_records) if max_records else total_len
    records = []
    
    for i in range(limit):
        try:
            row = split[i]
            rec = {
                "record_id": i,
                "timestamp": row.get("start_time"),
                "end_time": row.get("end_time"),
                "sampling_rate": row.get("sampling_rate", 10),
            }
            
            # Extract categorical labels safely
            labels = row.get("labels") or {}
            rec["zone"] = str(labels.get("zone", "Unknown"))
            rec["application"] = str(labels.get("application", "Unknown"))
            rec["mobility"] = str(labels.get("mobility", "Unknown"))
            rec["congestion"] = str(labels.get("congestion", "No"))
            rec["anomaly_present"] = str(labels.get("anomaly_present", "No"))
            
            # Extract anomaly metadata
            anom = row.get("anomalies") or {}
            anom_type = anom.get("type", "")
            rec["anomaly_type"] = anom_type if (anom_type and str(anom_type).lower() != "none") else "Normal"
            rec["anomaly_exists"] = bool(anom.get("exists", False))
            aff = anom.get("affected_kpis") or []
            rec["affected_kpis"] = ", ".join([str(k) for k in aff if k])
            rec["troubleshooting_summary"] = str(anom.get("troubleshooting_tickets", ""))
            
            # Extract numeric KPI statistics (mean & standard deviation)
            stats = row.get("statistics") or {}
            if isinstance(stats, dict):
                for kpi_name, stat_dict in stats.items():
                    if isinstance(stat_dict, dict):
                        m_val = stat_dict.get("mean", np.nan)
                        v_val = stat_dict.get("variance", 0.0)
                        rec[kpi_name] = m_val
                        rec[f"{kpi_name}_std"] = float(np.sqrt(max(0.0, v_val))) if v_val is not None else 0.0
            
            records.append(rec)
        except Exception:
            continue
            
    df = pd.DataFrame(records)
    
    # Compute derived standard telecom operational metrics if components exist
    if "TX_Bytes" in df.columns and "RX_Bytes" in df.columns:
        df["Throughput_Total_KB"] = (df["TX_Bytes"] + df["RX_Bytes"]) / 1024.0
    if "DL_BLER" in df.columns:
        df["Packet_Loss_Rate_Pct"] = df["DL_BLER"] * 100.0
        
    return df


def get_record_timeseries(ds_or_split, record_idx: int) -> Optional[pd.DataFrame]:
    """
    Extract raw high-frequency time-series array for all KPIs in a single chunk.
    Allows engineers to inspect micro-bursts and waveforms across the chunk window.
    """
    if ds_or_split is None:
        return None
        
    split = ds_or_split["full"] if hasattr(ds_or_split, "__getitem__") and "full" in ds_or_split else ds_or_split
    if record_idx < 0 or record_idx >= len(split):
        return None
        
    try:
        sample = split[record_idx]
        kpis = sample.get("KPIs", {})
        if not isinstance(kpis, dict) or not kpis:
            return None
            
        ts_df = pd.DataFrame(kpis)
        ts_df["step"] = range(len(ts_df))
        return ts_df
    except Exception:
        return None
