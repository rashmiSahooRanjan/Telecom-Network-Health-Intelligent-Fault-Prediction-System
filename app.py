"""
Telecom Network Health & Intelligent Fault Prediction System
Main Streamlit Application
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration FIRST before any other Streamlit calls
st.set_page_config(
    page_title="Telecom Network Health & Fault Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular project components
from src.utils import (
    apply_custom_theme, render_banner, render_disclaimer,
    format_bytes, get_status_class, download_csv_button
)
from src.data_loader import (
    load_telecom_dataset_hf, load_sample_dataset,
    inspect_dataset, extract_kpi_dataframe, get_record_timeseries
)
from src.preprocessing import (
    clean_dataset, detect_kpi_columns, validate_uploaded_csv,
    preprocess_for_ml
)
from src.feature_engineering import engineer_telecom_features
from src.network_health import (
    calculate_single_health_score, compute_dataset_health,
    DEFAULT_HEALTH_THRESHOLDS
)
from src.anomaly_detection import detect_anomalies, get_anomaly_kpi_comparison
from src.model_training import run_supervised_pipeline
from src.prediction import predict_single_sample
from src.recommendations import analyze_faults_and_causes, generate_recommendations
from src.eda import (
    get_kpi_summary_statistics, plot_kpi_distribution, plot_kpi_time_series,
    plot_kpi_correlation_matrix, plot_kpi_boxplot, plot_kpi_scatter,
    plot_raw_chunk_waveform
)

# Apply NOC dark/clean visual styling
apply_custom_theme()


# ==============================================================================
# STATE & DATA INITIALIZATION
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_base_data(source_mode: str, sample_size: int = 2000):
    """Load and preprocess dataset based on selected sidebar source mode."""
    if source_mode == "Fast Sample (Instant)":
        df_raw = load_sample_dataset()
        if df_raw.empty:
            # Fallback to HF if sample csv not present
            ds, _ = load_telecom_dataset_hf()
            df_raw = extract_kpi_dataframe(ds, max_records=sample_size)
    else:
        ds, msg = load_telecom_dataset_hf()
        if ds is not None:
            df_raw = extract_kpi_dataframe(ds, max_records=sample_size)
        else:
            df_raw = load_sample_dataset()

    cleaned = clean_dataset(df_raw)
    engineered = engineer_telecom_features(cleaned)
    return engineered


# Sidebar navigation & configuration
st.sidebar.markdown("### 📡 Telecom NOC Navigation")
page = st.sidebar.radio(
    "Select System Module:",
    [
        "1. Home",
        "2. Dataset Overview",
        "3. Network KPI Dashboard",
        "4. EDA",
        "5. Anomaly Detection",
        "6. ML Prediction",
        "7. Network Health",
        "8. Fault Analysis",
        "9. Upload KPI Data",
        "10. Reports",
        "11. About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Data Source Configuration")

data_source = st.sidebar.selectbox(
    "Dataset Provider:",
    ["Fast Sample (Instant)", "Live Hugging Face (Full)"],
    index=0,
    help="Fast Sample provides instant loading with all authentic anomaly cases. Live HF loads directly from Hugging Face repository."
)

hf_sample_size = 2000
if data_source == "Live Hugging Face (Full)":
    hf_sample_size = st.sidebar.slider(
        "Max HF Records to Process:",
        min_value=500,
        max_value=10000,
        value=2500,
        step=500
    )

# Active dataset loading
with st.spinner("Loading telemetry dataset..."):
    # Check if custom uploaded data exists in session state
    if "custom_df" in st.session_state and st.session_state["custom_df"] is not None:
        df = st.session_state["custom_df"]
        active_source_label = "Custom Uploaded CSV"
    else:
        df = get_base_data(data_source, sample_size=hf_sample_size)
        active_source_label = f"TelecomTS ({data_source})"

# Ensure health scores are computed
if "Health_Score" not in df.columns and not df.empty:
    df = compute_dataset_health(df)

# Sidebar System Health Status Summary
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📊 Active Stream Status")
if not df.empty:
    avg_health = df["Health_Score"].mean() if "Health_Score" in df.columns else 100.0
    st.sidebar.metric("Loaded Records", f"{len(df):,}")
    st.sidebar.metric("Average Health Score", f"{avg_health:.1f}/100")
    st.sidebar.info(f"**Source:** {active_source_label}")
else:
    st.sidebar.warning("No records loaded.")

st.sidebar.markdown(
    "<small style='color: #64748b;'>Prototype v1.0.0 • Python 3.10 • Streamlit Cloud Ready</small>",
    unsafe_allow_html=True
)


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if page == "1. Home":
    render_banner(
        title="Telecom Network Health & Intelligent Fault Prediction System",
        subtitle="Real-time KPI telemetry monitoring, anomaly detection, health scoring, and predictive fault diagnostics."
    )
    render_disclaimer()

    # KPI summary cards
    if not df.empty:
        kpi_cols = detect_kpi_columns(df)
        total_records = len(df)
        num_kpis = len(kpi_cols)
        
        # Count anomalies
        if "anomaly_exists" in df.columns:
            anomalies_count = int(df["anomaly_exists"].sum())
        elif "anomaly_present" in df.columns:
            anomalies_count = int((df["anomaly_present"].str.lower() == "yes").sum())
        else:
            anomalies_count = 0
            
        critical_count = int((df["Health_Score"] < 50.0).sum()) if "Health_Score" in df.columns else 0
        avg_score = float(df["Health_Score"].mean()) if "Health_Score" in df.columns else 100.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Records", f"{total_records:,}")
        with c2:
            st.metric("Telecom KPIs", f"{num_kpis}")
        with c3:
            st.metric("Anomalies Identified", f"{anomalies_count:,}", delta=f"{anomalies_count/total_records*100:.1f}% rate", delta_color="inverse")
        with c4:
            st.metric("Critical Health Records", f"{critical_count:,}", delta=f"{critical_count/total_records*100:.1f}%", delta_color="inverse")
        with c5:
            st.metric("Avg Network Health", f"{avg_score:.1f}/100")

    st.markdown("### 🌐 Operational Overview & Key Capabilities")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown(
            """
            This application is an end-to-end intelligent network telemetry analytics and fault prediction 
            system built for **Telecom Network Analyst Engineers**. It ingests multidimensional 5G radio and 
            transport KPIs from the open-source **TelecomTS** benchmark dataset, performs automated feature 
            normalization, detects abnormal network behavior using unsupervised machine learning, evaluates 
            multivariate health scores, and recommends actionable troubleshooting interventions.
            
            #### 🛠️ Core Capabilities:
            - **Hugging Face TelecomTS Integration:** Direct streaming and caching of 32,000 multi-zone 5G records.
            - **Telecom KPI Analytics:** Evaluation of RSRP, SNR, Downlink/Uplink BLER, PRB Utilization, Throughput, and Buffer delays.
            - **Unsupervised Anomaly Detection:** Isolation Forest algorithm detecting unexpected network deviations with configurable contamination.
            - **Multi-Model Supervised Prediction:** Comparative benchmarking of Logistic Regression, Random Forest, and Gradient Boosting.
            - **Transparent Health Scoring:** Fully inspectable 0-100 score with configurable engineering penalty thresholds.
            - **Domain-Specific Troubleshooting Engine:** Rule-based diagnostic mapping symptoms to root cause hypotheses.
            - **Custom Data Upload:** Seamless validation and analysis of arbitrary user-uploaded CSV telemetry.
            """
        )

    with col_b:
        st.markdown("#### ⚡ Network Health Distribution")
        if not df.empty and "Health_Status" in df.columns:
            status_counts = df["Health_Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            color_map = {
                "Healthy": "#22c55e",
                "Good": "#38bdf8",
                "Warning": "#eab308",
                "Critical": "#ef4444"
            }
            fig_pie = px.pie(
                status_counts,
                values="Count",
                names="Status",
                color="Status",
                color_discrete_map=color_map,
                hole=0.45
            )
            fig_pie.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ==============================================================================
# PAGE 2: DATASET OVERVIEW
# ==============================================================================
elif page == "2. Dataset Overview":
    render_banner(
        title="Dataset Overview & Schema Inspection",
        subtitle="Inspect raw telemetry structures, data types, missing records, and dataset distributions."
    )
    render_disclaimer()

    if df.empty:
        st.error("No dataset currently available.")
    else:
        kpi_cols = detect_kpi_columns(df)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Total Columns", f"{len(df.columns)}")
        m3.metric("Detected KPIs", f"{len(kpi_cols)}")
        m4.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
        m5.metric("Duplicate Rows", f"{df.duplicated().sum()}")

        st.markdown("### 📋 Processed Telemetry Preview")
        st.dataframe(df.head(25), use_container_width=True)

        download_csv_button(df, "telecom_network_kpis_processed.csv", "📥 Download Processed Dataset (CSV)")

        t1, t2 = st.tabs(["📊 Column Metadata & Data Types", "🔍 Hugging Face Raw Schema Details"])
        with t1:
            meta_df = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": [str(t) for t in df.dtypes],
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
                "Unique Values": [df[c].nunique() for c in df.columns]
            })
            st.dataframe(meta_df, use_container_width=True)

        with t2:
            st.markdown(
                """
                **Hugging Face Dataset Identifier:** `AliMaatouk/TelecomTS`  
                **Split Structure:** `full` split containing 33 chunked JSONL records across 3 zones (Zone A, B, C, and In-Motion).  
                **Key Nested Structures:**
                - `KPIs`: High-resolution 128-sample timeseries array for each chunk.
                - `statistics`: Pre-calculated mean, variance, trend, and periodicity per KPI.
                - `labels`: Application (File, YouTube, Twitch), mobility, congestion, anomaly flags.
                - `anomalies`: Anomaly type, affected KPI tags, and troubleshooting diagnostic tickets.
                """
            )


# ==============================================================================
# PAGE 3: NETWORK KPI DASHBOARD
# ==============================================================================
elif page == "3. Network KPI Dashboard":
    render_banner(
        title="Network KPI Performance Dashboard",
        subtitle="Real-time multi-metric exploration, statistical aggregates, and micro-waveform inspection."
    )

    if df.empty:
        st.warning("Please load telemetry data first.")
    else:
        kpi_cols = detect_kpi_columns(df)
        
        # Primary KPI selection
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            selected_kpi = st.selectbox("Primary KPI to Analyze:", kpi_cols, index=0)
        with col_ctrl2:
            category_group = st.selectbox(
                "Group/Color Visualizations By:",
                [None, "zone", "application", "congestion", "anomaly_type", "Health_Status"],
                index=1
            )

        # Statistical summary cards for selected KPI
        if selected_kpi in df.columns:
            s_mean = df[selected_kpi].mean()
            s_min = df[selected_kpi].min()
            s_max = df[selected_kpi].max()
            s_med = df[selected_kpi].median()
            s_std = df[selected_kpi].std()

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Average", f"{s_mean:.3f}")
            k2.metric("Minimum", f"{s_min:.3f}")
            k3.metric("Maximum", f"{s_max:.3f}")
            k4.metric("Median", f"{s_med:.3f}")
            k5.metric("Std Dev", f"{s_std:.3f}")

        # Interactive charts
        tab_dist, tab_time, tab_box, tab_corr, tab_wave = st.tabs([
            "📊 Distribution", "📈 Trend Over Time", "📦 Box & Outlier",
            "🔗 KPI Correlations", "🌊 Micro-Waveform Trace"
        ])

        with tab_dist:
            fig_dist = plot_kpi_distribution(df, selected_kpi, color_by=category_group)
            st.plotly_chart(fig_dist, use_container_width=True)

        with tab_time:
            fig_time = plot_kpi_time_series(df, selected_kpi, color_by=category_group)
            st.plotly_chart(fig_time, use_container_width=True)

        with tab_box:
            fig_box = plot_kpi_boxplot(df, [selected_kpi], group_by=category_group)
            st.plotly_chart(fig_box, use_container_width=True)

        with tab_corr:
            fig_corr = plot_kpi_correlation_matrix(df, kpi_cols[:12])
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab_wave:
            st.markdown("#### High-Resolution Chunk Waveform Viewer")
            st.info("TelecomTS captures 128 high-frequency telemetry steps per observation window. Inspect micro-bursts for any record:")
            rec_id = st.number_input("Record ID to inspect waveform:", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
            
            # Load raw sample for waveform
            ds_raw, _ = load_telecom_dataset_hf()
            if ds_raw:
                ts_df = get_record_timeseries(ds_raw, rec_id)
                if ts_df is not None and selected_kpi in ts_df.columns:
                    fig_wave = plot_raw_chunk_waveform(ts_df, selected_kpi)
                    st.plotly_chart(fig_wave, use_container_width=True)
                else:
                    st.warning(f"Waveform array for '{selected_kpi}' not found in record {rec_id}.")
            else:
                st.warning("Live waveform viewer requires connection to Hugging Face dataset.")


# ==============================================================================
# PAGE 4: EDA
# ==============================================================================
elif page == "4. EDA":
    render_banner(
        title="Exploratory Data Analysis (EDA)",
        subtitle="Multivariate statistical analysis, distributions, categorical splits, and outlier profiles."
    )

    if df.empty:
        st.warning("No data available.")
    else:
        kpi_cols = detect_kpi_columns(df)

        st.markdown("### 📈 Comprehensive Telecom KPI Statistical Summary")
        stats_summary = get_kpi_summary_statistics(df, kpi_cols)
        st.dataframe(stats_summary, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Bivariate Cross-KPI Relationship")
            x_kpi = st.selectbox("X-Axis KPI:", kpi_cols, index=0)
            y_kpi = st.selectbox("Y-Axis KPI:", kpi_cols, index=min(2, len(kpi_cols)-1))
            color_sel = st.selectbox("Color By:", [None, "zone", "application", "congestion", "anomaly_type"], index=1)
            fig_scatter = plot_kpi_scatter(df, x_kpi, y_kpi, color_by=color_sel)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_right:
            st.markdown("#### Categorical Operational Profiles")
            cat_choice = st.selectbox("Categorical Feature:", ["zone", "application", "congestion", "mobility"], index=0)
            if cat_choice in df.columns:
                counts = df[cat_choice].value_counts().reset_index()
                counts.columns = [cat_choice, "Count"]
                fig_cat = px.bar(
                    counts, x=cat_choice, y="Count", color=cat_choice,
                    title=f"Distribution of Telemetry Records by {cat_choice.title()}",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("### 📦 Multi-KPI Outlier & Distribution Boxplots")
        selected_box_kpis = st.multiselect(
            "Select KPIs for Boxplot Comparison:",
            kpi_cols,
            default=kpi_cols[:4] if len(kpi_cols) >= 4 else kpi_cols
        )
        if selected_box_kpis:
            fig_multi_box = plot_kpi_boxplot(df, selected_box_kpis)
            st.plotly_chart(fig_multi_box, use_container_width=True)


# ==============================================================================
# PAGE 5: ANOMALY DETECTION
# ==============================================================================
elif page == "5. Anomaly Detection":
    render_banner(
        title="Unsupervised Anomaly Detection",
        subtitle="Isolation Forest ML pipeline detecting multivariate telemetry drift and network outliers."
    )

    if df.empty:
        st.warning("No data available.")
    else:
        kpi_cols = detect_kpi_columns(df)

        with st.expander("🛠️ Isolation Forest Hyperparameters & Feature Selection", expanded=True):
            col_ad1, col_ad2 = st.columns([1, 2])
            with col_ad1:
                contamination = st.slider(
                    "Expected Contamination Rate:",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    step=0.01,
                    help="Proportion of outliers in the data set."
                )
            with col_ad2:
                selected_ad_features = st.multiselect(
                    "Telemetry Features for Anomaly Model:",
                    kpi_cols,
                    default=kpi_cols[:8] if len(kpi_cols) >= 8 else kpi_cols
                )

        if not selected_ad_features:
            st.error("Please select at least 2 KPI features.")
        else:
            with st.spinner("Training Isolation Forest and scoring anomalies..."):
                ad_df, ad_summary = detect_anomalies(df, selected_ad_features, contamination=contamination)

            # Metric display
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records Evaluated", f"{ad_summary['total_records']:,}")
            c2.metric("Detected Anomalies", f"{ad_summary['anomaly_count']:,}", delta=f"{ad_summary['anomaly_rate_pct']}% rate", delta_color="inverse")
            c3.metric("Normal Records", f"{ad_summary['normal_count']:,}")
            c4.metric("Contamination Setting", f"{contamination*100:.1f}%")

            # Visualizations
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_score = px.histogram(
                    ad_df,
                    x="anomaly_score",
                    color="anomaly_label",
                    color_discrete_map={"Normal": "#22c55e", "Anomaly": "#ef4444"},
                    title="Isolation Forest Decision Function Score Distribution",
                    template="plotly_dark",
                    nbins=40
                )
                st.plotly_chart(fig_score, use_container_width=True)

            with col_v2:
                # 2D scatter of top two KPIs showing anomalies
                if len(selected_ad_features) >= 2:
                    fig_ad_scatter = px.scatter(
                        ad_df,
                        x=selected_ad_features[0],
                        y=selected_ad_features[1],
                        color="anomaly_label",
                        color_discrete_map={"Normal": "#22c55e", "Anomaly": "#ef4444"},
                        title=f"Detected Anomalies: {selected_ad_features[0]} vs {selected_ad_features[1]}",
                        template="plotly_dark",
                        opacity=0.8
                    )
                    st.plotly_chart(fig_ad_scatter, use_container_width=True)

            st.markdown("### 📊 Mean KPI Drift: Normal vs Anomalous Records")
            drift_df = get_anomaly_kpi_comparison(ad_df, selected_ad_features)
            st.dataframe(drift_df, use_container_width=True)

            st.markdown("### 🚨 Detected Anomalous Records Sample")
            anom_records = ad_df[ad_df["is_anomaly"] == 1]
            st.dataframe(anom_records.head(50), use_container_width=True)
            download_csv_button(anom_records, "detected_network_anomalies.csv", "📥 Download Anomaly Records (CSV)")


# ==============================================================================
# PAGE 6: ML PREDICTION
# ==============================================================================
elif page == "6. ML Prediction":
    render_banner(
        title="Supervised Machine Learning & Fault Prediction",
        subtitle="Model benchmarking, confusion matrices, feature importances, and interactive real-time inference."
    )

    if df.empty:
        st.warning("No data available.")
    else:
        kpi_cols = detect_kpi_columns(df)

        tab_train, tab_infer = st.tabs(["🤖 Model Training & Comparison", "🔮 Real-Time Fault Predictor"])

        with tab_train:
            st.markdown("### 🏋️ Multi-Model Supervised Training Pipeline")
            
            # Select target
            possible_targets = [c for c in ["anomaly_present", "congestion", "anomaly_type"] if c in df.columns]
            if not possible_targets:
                possible_targets = ["Health_Status"]

            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                target_choice = st.selectbox("Target Classification Label:", possible_targets, index=0)
            with col_m2:
                ml_features = st.multiselect(
                    "Features to include:",
                    kpi_cols,
                    default=kpi_cols[:8] if len(kpi_cols) >= 8 else kpi_cols
                )

            if st.button("🚀 Train & Compare Models", type="primary"):
                with st.spinner("Training Logistic Regression, Random Forest, and Gradient Boosting..."):
                    ml_results = run_supervised_pipeline(df, ml_features, target_choice)
                    st.session_state["ml_pipeline"] = ml_results

            if "ml_pipeline" in st.session_state:
                res = st.session_state["ml_pipeline"]
                
                st.markdown("#### 🏆 Model Performance Benchmark Table")
                st.dataframe(res["comparison_table"], use_container_width=True)
                st.success(f"**Best Model Selected:** {res['best_model_name']} (F1-Score: {res['best_f1_score']:.4f})")

                # Confusion Matrix of Best Model
                best_clf_info = res["model_results"].get(res["best_model_name"], {})
                cm = best_clf_info.get("confusion_matrix")
                if cm is not None:
                    col_cm, col_fi = st.columns(2)
                    with col_cm:
                        st.markdown(f"#### Confusion Matrix ({res['best_model_name']})")
                        classes = [str(c) for c in res["classes"]]
                        fig_cm = px.imshow(
                            cm,
                            x=classes,
                            y=classes,
                            text_auto=True,
                            labels=dict(x="Predicted Condition", y="True Condition"),
                            title="Confusion Matrix",
                            template="plotly_dark",
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with col_fi:
                        st.markdown("#### Key Feature Importances")
                        fi_dict = res.get("feature_importances", {})
                        if fi_dict:
                            fi_df = pd.DataFrame(list(fi_dict.items()), columns=["Feature", "Importance"])
                            fig_fi = px.bar(
                                fi_df.head(10),
                                x="Importance",
                                y="Feature",
                                orientation="h",
                                title="Top Predictive Telecom KPIs",
                                template="plotly_dark",
                                color="Importance",
                                color_continuous_scale="Viridis"
                            )
                            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig_fi, use_container_width=True)

        with tab_infer:
            st.markdown("### 🔮 Interactive Telecom KPI Fault Inference")
            st.markdown("Simulate network operational conditions by adjusting key 5G telemetry inputs:")

            if "ml_pipeline" not in st.session_state:
                st.info("Please train models in the 'Model Training & Comparison' tab first, or use default parameters below.")
                if st.button("Initialize Fast Default Model"):
                    default_target = "anomaly_present" if "anomaly_present" in df.columns else "Health_Status"
                    st.session_state["ml_pipeline"] = run_supervised_pipeline(df, kpi_cols[:8], default_target)
                    st.rerun()

            if "ml_pipeline" in st.session_state:
                pipe = st.session_state["ml_pipeline"]
                active_model = pipe["model_results"][pipe["best_model_name"]]["model"]
                scaler = pipe["scaler"]
                f_cols = pipe["feature_cols"]
                encoder = pipe["label_encoder"]

                # Generate dynamic sliders based on dataset statistics
                input_vals = {}
                cols = st.columns(min(3, len(f_cols)))
                for idx, col_name in enumerate(f_cols):
                    with cols[idx % len(cols)]:
                        c_min = float(df[col_name].min()) if col_name in df.columns else 0.0
                        c_max = float(df[col_name].max()) if col_name in df.columns else 100.0
                        c_mean = float(df[col_name].mean()) if col_name in df.columns else 50.0
                        
                        input_vals[col_name] = st.slider(
                            f"{col_name}:",
                            min_value=round(c_min, 2),
                            max_value=round(c_max, 2),
                            value=round(c_mean, 2)
                        )

                if st.button("⚡ Run Fault & Health Prediction", type="primary"):
                    pred_res = predict_single_sample(
                        input_vals, active_model, scaler, f_cols, label_encoder=encoder
                    )

                    st.markdown("---")
                    st.markdown("### 📋 Real-Time Diagnostic Output")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Predicted Condition", pred_res["predicted_label"])
                    if pred_res["confidence"]:
                        r2.metric("Confidence / Probability", f"{pred_res['confidence']*100:.1f}%")
                    else:
                        r2.metric("Confidence", "Deterministic")
                    r3.metric("Risk Level", pred_res["risk_level"])
                    r4.metric("Network Health Score", f"{pred_res['health_score']}/100", pred_res["health_status"])

                    col_diag1, col_diag2 = st.columns(2)
                    with col_diag1:
                        st.markdown("#### ⚖️ Contributing Penalties & Reasons")
                        if pred_res["penalties"]:
                            for k, v in pred_res["penalties"].items():
                                st.write(f"- **{k}:** -{v:.0f} pts penalty")
                            for r in pred_res["contributing_reasons"]:
                                st.warning(r)
                        else:
                            st.success("All analyzed metrics are within healthy limits.")

                    with col_diag2:
                        st.markdown("#### 🛠️ Recommended Troubleshooting Steps")
                        for rec in pred_res["recommendations"]:
                            st.info(f"**{rec.get('category')}:** {rec.get('recommended_action', rec.get('action'))}")


# ==============================================================================
# PAGE 7: NETWORK HEALTH
# ==============================================================================
elif page == "7. Network Health":
    render_banner(
        title="Network Health Score Engine",
        subtitle="Transparent 0-100 composite health scoring with configurable SLA penalty thresholds."
    )

    if df.empty:
        st.warning("No data available.")
    else:
        with st.expander("⚙️ Configure Health Score Penalty Thresholds (Custom Engineering Rules)", expanded=False):
            st.markdown(
                """
                *Note: These thresholds are project-defined demonstrative engineering benchmarks, 
                not official 3GPP/ITU standards.*
                """
            )
            ct1, ct2, ct3 = st.columns(3)
            with ct1:
                th_bler_warn = st.slider("BLER Warning Threshold:", 0.01, 0.20, DEFAULT_HEALTH_THRESHOLDS["bler_warning"], 0.01)
                th_bler_crit = st.slider("BLER Critical Threshold:", 0.05, 0.40, DEFAULT_HEALTH_THRESHOLDS["bler_critical"], 0.01)
            with ct2:
                th_rsrp_warn = st.slider("RSRP Warning (dBm):", -120.0, -80.0, DEFAULT_HEALTH_THRESHOLDS["rsrp_warning"], 1.0)
                th_rsrp_crit = st.slider("RSRP Critical (dBm):", -130.0, -90.0, DEFAULT_HEALTH_THRESHOLDS["rsrp_critical"], 1.0)
            with ct3:
                th_prb_warn = st.slider("PRB Util Warning (%):", 50.0, 90.0, DEFAULT_HEALTH_THRESHOLDS["prb_util_warning"], 5.0)
                th_prb_crit = st.slider("PRB Util Critical (%):", 70.0, 99.0, DEFAULT_HEALTH_THRESHOLDS["prb_util_critical"], 5.0)

            custom_thresholds = {
                **DEFAULT_HEALTH_THRESHOLDS,
                "bler_warning": th_bler_warn,
                "bler_critical": th_bler_crit,
                "rsrp_warning": th_rsrp_warn,
                "rsrp_critical": th_rsrp_crit,
                "prb_util_warning": th_prb_warn,
                "prb_util_critical": th_prb_crit
            }

        # Compute with active thresholds
        health_df = compute_dataset_health(df, custom_thresholds)

        # Overview Metrics
        avg_score = health_df["Health_Score"].mean()
        h_status = health_df["Health_Status"].value_counts()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Health Score", f"{avg_score:.1f}/100")
        c2.metric("Healthy Records (90-100)", f"{h_status.get('Healthy', 0):,}")
        c3.metric("Warning Records (50-69)", f"{h_status.get('Warning', 0):,}")
        c4.metric("Critical Records (0-49)", f"{h_status.get('Critical', 0):,}", delta_color="inverse")

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            fig_hist = px.histogram(
                health_df,
                x="Health_Score",
                color="Health_Status",
                color_discrete_map={
                    "Healthy": "#22c55e",
                    "Good": "#38bdf8",
                    "Warning": "#eab308",
                    "Critical": "#ef4444"
                },
                nbins=25,
                title="Network Health Score Distribution Across All Records",
                template="plotly_dark"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_g2:
            st.markdown("#### Health Score Category Legend")
            st.markdown(
                """
                - 🟢 **90 – 100 | Healthy:** Operating within nominal engineering margins.
                - 🔵 **70 – 89 | Good:** Minor degradation or moderate channel fading.
                - 🟡 **50 – 69 | Warning:** Elevated BLER, buffer backlogs, or heavy PRB load.
                - 🔴 **0 – 49 | Critical:** Severe RF jamming, buffer overflows, or drop-call conditions.
                """
            )
            # Health Score vs Throughput
            if "Throughput_Total_KB" in health_df.columns:
                fig_ht = px.scatter(
                    health_df.sample(min(len(health_df), 1500), random_state=42),
                    x="Health_Score",
                    y="Throughput_Total_KB",
                    color="Health_Status",
                    color_discrete_map={"Healthy": "#22c55e", "Good": "#38bdf8", "Warning": "#eab308", "Critical": "#ef4444"},
                    title="Health Score vs Total Throughput",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_ht, use_container_width=True)

        st.markdown("### ⚠️ Highest Risk Records Requiring Investigation")
        high_risk_df = health_df.sort_values(by="Health_Score", ascending=True)
        st.dataframe(high_risk_df.head(25), use_container_width=True)


# ==============================================================================
# PAGE 8: FAULT ANALYSIS
# ==============================================================================
elif page == "8. Fault Analysis":
    render_banner(
        title="Telecom Fault Analysis & Root Cause Hypotheses",
        subtitle="Transparent domain-rule inference engine detecting congestion, jamming, and link failures."
    )

    if df.empty:
        st.warning("No data available.")
    else:
        st.markdown(
            """
            #### ⚙️ Expert Rule Engine Architecture
            The fault analyzer inspects multidimensional telemetry using telecom domain correlation rules:
            - **IF** Downlink PRB Utilization > 85% **AND** Uplink Buffer Backlog > 25,000 B  
              ➔ **Possible Cause:** Network Congestion & Buffer Overflow.
            - **IF** SNR < 8 dB **AND** DL BLER > 10% with passable RSRP  
              ➔ **Possible Cause:** RF Interference or Active Jamming.
            - **IF** RSRP < -110 dBm **AND** SNR < 5 dB  
              ➔ **Possible Cause:** Weak Coverage / Cell Edge Boundary.
            - **IF** DL BLER > 15%  
              ➔ **Possible Cause:** High Link Block Error Rate & Retransmissions.
            """
        )
        st.caption("*All identified causes represent probable hypotheses based on telemetry, not confirmed hardware causes.*")

        st.markdown("### 🔍 Live Record Fault Investigator")
        sel_rec = st.number_input("Select Record Index for Deep-Dive Fault Inspection:", 0, len(df)-1, 0)
        rec_data = df.iloc[sel_rec]

        faults = analyze_faults_and_causes(rec_data)

        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            st.markdown(f"#### Record #{sel_rec} Telemetry")
            st.write(f"- **Zone:** {rec_data.get('zone', 'Unknown')}")
            st.write(f"- **Application:** {rec_data.get('application', 'Unknown')}")
            st.write(f"- **RSRP:** {rec_data.get('RSRP', 'N/A')} dBm")
            st.write(f"- **SNR:** {rec_data.get('UL_SNR', 'N/A')} dB")
            st.write(f"- **DL BLER:** {rec_data.get('DL_BLER', 'N/A')}")
            st.write(f"- **PRB Util DL:** {rec_data.get('PRB_Utilization_DL', 'N/A')}%")
            st.write(f"- **Buffer Backlog:** {rec_data.get('Estimated_UL_Buffer', 'N/A')} B")

        with col_f2:
            st.markdown("#### 📋 Fault Diagnostics & Recommendations")
            if not faults:
                st.success("No active faults or abnormal threshold breaches detected in this record.")
            else:
                for f in faults:
                    st.error(f"**Detected Fault:** {f.get('category')} (Severity: {f.get('severity')})")
                    st.write(f"**Evidence:** {f.get('evidence')}")
                    st.write(f"**Possible Cause:** {f.get('possible_cause')}")
                    st.info(f"**Recommended Next Step:** {f.get('recommended_action')}")

        if "troubleshooting_summary" in rec_data and rec_data["troubleshooting_summary"]:
            with st.expander("🎫 Dataset Troubleshooting Ticket for this Record", expanded=True):
                st.markdown(rec_data["troubleshooting_summary"])


# ==============================================================================
# PAGE 9: UPLOAD KPI DATA
# ==============================================================================
elif page == "9. Upload KPI Data":
    render_banner(
        title="Upload Custom KPI Telemetry",
        subtitle="Ingest external CSV telemetry, auto-validate features, and run full network health analysis."
    )

    uploaded_file = st.file_uploader("Select a Telecom KPI CSV file:", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            is_valid, msg, detected_kpis = validate_uploaded_csv(uploaded_df)
            
            if not is_valid:
                st.error(msg)
            else:
                st.success(f"File validated successfully! {msg}")

                st.markdown("#### Detected Telemetry Columns:")
                selected_user_kpis = st.multiselect(
                    "Confirm / Select KPI columns to include:",
                    detected_kpis,
                    default=detected_kpis
                )

                if st.button("⚡ Ingest & Analyze Uploaded Data", type="primary"):
                    cleaned_user_df = clean_dataset(uploaded_df)
                    engineered_user_df = engineer_telecom_features(cleaned_user_df)
                    scored_user_df = compute_dataset_health(engineered_user_df)
                    
                    st.session_state["custom_df"] = scored_user_df
                    st.success("Uploaded dataset successfully activated across the entire dashboard!")
                    st.rerun()

                st.markdown("### Preview of Uploaded CSV")
                st.dataframe(uploaded_df.head(15), use_container_width=True)

        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

    if "custom_df" in st.session_state and st.session_state["custom_df"] is not None:
        st.markdown("---")
        st.info("A custom uploaded dataset is currently active.")
        if st.button("🔄 Reset to Default TelecomTS Dataset"):
            st.session_state["custom_df"] = None
            st.rerun()


# ==============================================================================
# PAGE 10: REPORTS
# ==============================================================================
elif page == "10. Reports":
    render_banner(
        title="Network Health & Executive Summary Reports",
        subtitle="Exportable network diagnostics, compliance summaries, and high-risk site records."
    )

    if df.empty:
        st.warning("No data available.")
    else:
        kpi_cols = detect_kpi_columns(df)
        total_records = len(df)
        avg_score = df["Health_Score"].mean() if "Health_Score" in df.columns else 100.0

        st.markdown("### 📊 Executive Telemetry Audit Summary")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records Inspected", f"{total_records:,}")
        c2.metric("Mean Network Health Score", f"{avg_score:.2f}/100")
        c3.metric("Critical Records Requiring NOC Action", f"{(df['Health_Score'] < 50.0).sum():,}")

        # Summary text generation
        report_md = f"""# Telecom Network Health Audit Report
**Generated Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluated Records:** {total_records:,}  
**Average Network Health Score:** {avg_score:.2f} / 100  

---
### Status Breakdown:
- **Healthy (90-100):** {(df['Health_Score'] >= 90).sum():,} records ({(df['Health_Score'] >= 90).mean()*100:.1f}%)
- **Good (70-89):** {((df['Health_Score'] >= 70) & (df['Health_Score'] < 90)).sum():,} records
- **Warning (50-69):** {((df['Health_Score'] >= 50) & (df['Health_Score'] < 70)).sum():,} records
- **Critical (0-49):** {(df['Health_Score'] < 50).sum():,} records

### Key KPI Benchmarks:
"""
        for k in kpi_cols[:6]:
            if k in df.columns:
                report_md += f"- **{k}:** Mean={df[k].mean():.2f}, Min={df[k].min():.2f}, Max={df[k].max():.2f}\n"

        st.markdown(report_md)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Executive Summary (Markdown)",
                data=report_md,
                file_name="telecom_health_report.md",
                mime="text/markdown"
            )
        with col_dl2:
            download_csv_button(df, "telecom_full_audit_records.csv", "📥 Download Complete Telemetry Data (CSV)")


# ==============================================================================
# PAGE 11: ABOUT PROJECT
# ==============================================================================
elif page == "11. About Project":
    render_banner(
        title="About the Project & Technical Interview Guide",
        subtitle="Architecture, methodology, public dataset provenance, and interview preparation answers."
    )
    render_disclaimer()

    st.markdown(
        """
        ### 🎯 How I Explain This Project in an Interview
        > *"I developed a Telecom Network Health and Intelligent Fault Prediction System using Python and Streamlit. 
        I used the public 5G telecom dataset TelecomTS containing multi-zone network KPI information. The system analyzes 
        network KPIs, detects abnormal behavior using unsupervised Isolation Forests, benchmarks supervised classification 
        models, calculates a transparent 0-100 network health score, and provides actionable troubleshooting suggestions. 
        I used Pandas for data manipulation, Scikit-learn for machine learning, Plotly for interactive dashboards, and Streamlit 
        for the web interface. This project gave me deep hands-on experience in how cellular network telemetry can be analyzed 
        to proactively detect faults before they disrupt user service."*

        ---
        ### 💬 Technical Interview Q&A Cheatsheet
        """
    )

    with st.expander("Q: Why did you choose this project and use TelecomTS?", expanded=True):
        st.markdown(
            """
            - **Why TelecomTS:** TelecomTS is a realistic open-source benchmark dataset containing actual 5G radio 
            telemetry (RSRP, BLER, MCS, PRB utilization, Buffer delays) recorded across multiple zones, mobility states, 
            and authentic anomaly scenarios (jamming, antenna failure, buffer overflow).
            - **Why this project:** Real telecom networks generate millions of telemetry records daily. Traditional manual 
            thresholding fails during subtle multi-parameter degradations. Applying machine learning allows automated 
            outlier detection and proactive root-cause analysis.
            """
        )

    with st.expander("Q: What is the difference between RSRP, SNR, BLER, and PRB Utilization?"):
        st.markdown(
            """
            - **RSRP (Reference Signal Received Power):** Measures cellular signal power received from the gNodeB (typically -125 dBm to -65 dBm).
            - **UL SNR (Signal-to-Noise Ratio):** Measures radio channel quality comparing desired signal against noise/interference (typically 0 to 30 dB).
            - **BLER (Block Error Rate):** Ratio of corrupted transport blocks to total received blocks. Standard SLA target is under 10%.
            - **PRB (Physical Resource Block) Utilization:** Percentage of available radio time-frequency slots occupied by traffic. >85% indicates cell congestion.
            """
        )

    with st.expander("Q: Why did you use Isolation Forest for Anomaly Detection?"):
        st.markdown(
            """
            - **Isolation Forest** works on the principle that anomalies are few and structurally different, making them easier 
            to isolate in random partition trees compared to normal points.
            - It does not assume normal distribution, handles multivariate telecom correlations seamlessly, scales linearly 
            in computational complexity `O(n)`, and operates completely unsupervised without requiring manual labels.
            """
        )

    with st.expander("Q: How does your Network Health Score work?"):
        st.markdown(
            """
            - It starts at a baseline of 100 points.
            - Transparent, domain-defined deductions are applied only when existing KPI telemetry breaches configurable warning/critical thresholds.
            - Deductions are allocated by failure mode: Block Error Rate (up to -30), Signal Quality (up to -25), PRB Congestion (up to -20), 
              Buffer backlog (up to -15), and Modulation degradation (up to -10).
            """
        )

    with st.expander("Q: What are the project limitations?"):
        st.markdown(
            """
            1. **Prototype Nature:** This is an analytical software prototype evaluated on public benchmark data.
            2. **No Direct Hardware Access:** It does not send write commands to live base stations or OSS/EMS networks.
            3. **Root Cause Hypotheses:** Fault classifications represent probabilistic hypotheses based on KPI correlations, not confirmed physical causes.
            """
        )
