"""
Telecom Network Health & Intelligent Fault Prediction System
EDA & Visualization Module: Interactive Plotly KPI distributions, correlation matrices, boxplots, and time-series.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional


PLOTLY_TEMPLATE = "plotly_dark"


def get_kpi_summary_statistics(df: pd.DataFrame, kpi_cols: List[str]) -> pd.DataFrame:
    """
    Generate professional telecom KPI statistics summary table:
    Mean, Std, Min, 25%, Median, 75%, Max for each available KPI.
    """
    if df.empty or not kpi_cols:
        return pd.DataFrame()
        
    valid_cols = [c for c in kpi_cols if c in df.columns]
    desc = df[valid_cols].describe().T
    
    # Rename and clean columns
    desc = desc.rename(columns={
        "mean": "Average",
        "std": "Std Dev",
        "min": "Minimum",
        "25%": "25th Pct",
        "50%": "Median",
        "75%": "75th Pct",
        "max": "Maximum"
    })
    
    # Format numeric values neatly
    return desc[["Average", "Std Dev", "Minimum", "Median", "Maximum"]].round(3)


def plot_kpi_distribution(df: pd.DataFrame, kpi_col: str, color_by: Optional[str] = None) -> go.Figure:
    """Plot interactive histogram with KDE / distribution overlay for a chosen KPI."""
    if df.empty or kpi_col not in df.columns:
        return go.Figure()
        
    plot_df = df.sample(min(len(df), 2500), random_state=42) if len(df) > 2500 else df
    
    color_param = color_by if (color_by and color_by in plot_df.columns) else None
    
    fig = px.histogram(
        plot_df,
        x=kpi_col,
        color=color_param,
        marginal="box",
        nbins=40,
        title=f"Distribution of {kpi_col}",
        template=PLOTLY_TEMPLATE,
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_kpi_time_series(df: pd.DataFrame, kpi_col: str, color_by: Optional[str] = None) -> go.Figure:
    """Plot KPI values chronologically across records/timestamps."""
    if df.empty or kpi_col not in df.columns:
        return go.Figure()
        
    plot_df = df.copy()
    x_axis = "timestamp" if "timestamp" in plot_df.columns and not plot_df["timestamp"].isnull().all() else "record_id"
    if x_axis not in plot_df.columns:
        plot_df["Index"] = range(len(plot_df))
        x_axis = "Index"
        
    # Downsample for smooth plotting if very large
    if len(plot_df) > 1500:
        plot_df = plot_df.iloc[::len(plot_df)//1500]

    color_param = color_by if (color_by and color_by in plot_df.columns) else None

    fig = px.line(
        plot_df,
        x=x_axis,
        y=kpi_col,
        color=color_param,
        title=f"Time Series Trend: {kpi_col}",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=3))
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_kpi_correlation_matrix(df: pd.DataFrame, kpi_cols: List[str]) -> go.Figure:
    """Generate interactive correlation heatmap for numeric KPI features."""
    if df.empty or not kpi_cols:
        return go.Figure()
        
    valid_cols = [c for c in kpi_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return go.Figure()

    corr = df[valid_cols].corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="KPI Pearson Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=PLOTLY_TEMPLATE
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_kpi_boxplot(df: pd.DataFrame, kpi_cols: List[str], group_by: Optional[str] = None) -> go.Figure:
    """Generate multi-KPI or grouped box plots for outlier analysis."""
    if df.empty or not kpi_cols:
        return go.Figure()
        
    valid_cols = [c for c in kpi_cols if c in df.columns]
    plot_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df

    if group_by and group_by in plot_df.columns and len(valid_cols) == 1:
        fig = px.box(
            plot_df,
            x=group_by,
            y=valid_cols[0],
            color=group_by,
            title=f"Outlier & Distribution Box Plot: {valid_cols[0]} by {group_by}",
            template=PLOTLY_TEMPLATE,
            points="outliers"
        )
    else:
        # Standardize for multi-box comparison
        melted = pd.melt(plot_df[valid_cols], var_name="KPI", value_name="Value")
        fig = px.box(
            melted,
            x="KPI",
            y="Value",
            color="KPI",
            title="KPI Distribution & Outlier Comparison",
            template=PLOTLY_TEMPLATE,
            points="outliers"
        )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_kpi_scatter(df: pd.DataFrame, x_kpi: str, y_kpi: str, color_by: Optional[str] = None) -> go.Figure:
    """Scatter comparison between two KPIs, optionally color-coded by condition/anomaly."""
    if df.empty or x_kpi not in df.columns or y_kpi not in df.columns:
        return go.Figure()
        
    plot_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df
    color_param = color_by if (color_by and color_by in plot_df.columns) else None

    fig = px.scatter(
        plot_df,
        x=x_kpi,
        y=y_kpi,
        color=color_param,
        title=f"KPI Cross-Comparison: {x_kpi} vs {y_kpi}",
        template=PLOTLY_TEMPLATE,
        opacity=0.75,
        hover_data=[c for c in ["record_id", "zone", "application", "anomaly_type"] if c in plot_df.columns]
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def plot_raw_chunk_waveform(ts_df: pd.DataFrame, selected_kpi: str) -> go.Figure:
    """Plot high-frequency time series waveform across the 128 samples of a single chunk."""
    if ts_df is None or ts_df.empty or selected_kpi not in ts_df.columns:
        return go.Figure()
        
    fig = px.line(
        ts_df,
        x="step",
        y=selected_kpi,
        title=f"Micro-Telemetry Waveform Trace: {selected_kpi} across Sample Chunks",
        template=PLOTLY_TEMPLATE
    )
    fig.update_traces(line_color="#06b6d4", line_width=2.5)
    fig.update_layout(
        xaxis_title="Time Step (within chunk)",
        yaxis_title=f"{selected_kpi} Value",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
