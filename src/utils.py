"""
Telecom Network Health & Intelligent Fault Prediction System
Utility functions: UI styling, metric formatting, CSV export, and reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime


def apply_custom_theme():
    """Inject modern, clean Telecom Network Operations Center (NOC) styling."""
    custom_css = """
    <style>
    /* Main container adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }
    
    /* Header branding styling */
    .telecom-brand {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .telecom-brand h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
    }
    .telecom-brand p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    .telecom-tag {
        display: inline-block;
        background: rgba(14, 165, 233, 0.2);
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Metric Cards */
    .kpi-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Health Status Badges */
    .badge-healthy {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-good {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(51, 65, 85, 0.4);
        border-left: 4px solid #0284c7;
        padding: 0.85rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_banner(title: str, subtitle: str, tag: str = "TelecomTS Public 5G Prototype"):
    """Render top page hero header banner."""
    st.markdown(
        f"""
        <div class="telecom-brand">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="telecom-tag">{tag}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Render the standard required public dataset prototype disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Prototype Notice:</strong> This project is an independent prototype developed using the public 
            open-source <strong>TelecomTS</strong> dataset (AliMaatouk/TelecomTS). It is not connected to Zenus Group's 
            internal network systems, live equipment, or proprietary telemetry. All KPI thresholds and health scores 
            are demonstrative analytics benchmarks.
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_bytes(bytes_val: float) -> str:
    """Format byte counts to human-readable strings (KB, MB, GB)."""
    if pd.isna(bytes_val) or bytes_val is None:
        return "N/A"
    try:
        val = float(bytes_val)
        if val >= 1024 * 1024 * 1024:
            return f"{val / (1024**3):.2f} GB"
        elif val >= 1024 * 1024:
            return f"{val / (1024**2):.2f} MB"
        elif val >= 1024:
            return f"{val / 1024:.2f} KB"
        else:
            return f"{val:.0f} B"
    except (ValueError, TypeError):
        return str(bytes_val)


def get_status_class(score: float) -> tuple[str, str]:
    """Map health score to category and color class."""
    if score >= 90:
        return "Healthy", "badge-healthy"
    elif score >= 70:
        return "Good", "badge-good"
    elif score >= 50:
        return "Warning", "badge-warning"
    else:
        return "Critical", "badge-critical"


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Download Dataset as CSV"):
    """Render standard Streamlit CSV download button."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}",
    )
