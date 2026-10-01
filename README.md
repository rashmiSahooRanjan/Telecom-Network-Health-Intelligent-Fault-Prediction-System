# 📡 Telecom Network Health & Intelligent Fault Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Hugging%20Face-TelecomTS-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
</p>

<p align="center">
  <strong>An end-to-end, production-grade 5G network telemetry analytics, anomaly detection, health scoring, and predictive fault classification system designed for Telecom Network Analyst Engineers.</strong>
</p>

---

> ### ⚠️ Project & Prototype Notice
> **Dataset Source:** This project is an independent analytics prototype developed using the public open-source **TelecomTS** dataset ([AliMaatouk/TelecomTS](https://huggingface.co/datasets/AliMaatouk/TelecomTS)) hosted on Hugging Face.  
> **Disclaimer:** This prototype is **not connected to Zenus Group's internal network systems, proprietary infrastructure, or live telecom equipment**. All KPI thresholds and health scoring benchmarks represent engineering demonstrative criteria.

---

## 📑 Table of Contents
- [1. Executive Summary & Industry Problem](#1-executive-summary--industry-problem)
- [2. System Architecture & Engineering Workflow](#2-system-architecture--engineering-workflow)
- [3. Key Platform Features & Dashboard Modules](#3-key-platform-features--dashboard-modules)
- [4. Telecom Domain Knowledge & KPI Telemetry](#4-telecom-domain-knowledge--kpi-telemetry)
- [5. Dataset Architecture: AliMaatouk/TelecomTS](#5-dataset-architecture-alimaatouktelecomts)
- [6. Machine Learning Pipelines](#6-machine-learning-pipelines)
  - [Unsupervised Anomaly Detection (Isolation Forest)](#unsupervised-anomaly-detection-isolation-forest)
  - [Supervised Multi-Model Fault Classification](#supervised-multi-model-fault-classification)
- [7. Transparent Network Health Scoring Engine](#7-transparent-network-health-scoring-engine)
- [8. Expert Rule-Based Troubleshooting Engine](#8-expert-rule-based-troubleshooting-engine)
- [9. Repository Directory Structure](#9-repository-directory-structure)
- [10. Quickstart & Local Installation](#10-quickstart--local-installation)
- [11. Streamlit Community Cloud Deployment Guide](#11-streamlit-community-cloud-deployment-guide)
- [12. Technical Interview Preparation Masterclass](#12-technical-interview-preparation-masterclass)
- [13. Engineering Limitations & Operational Boundaries](#13-engineering-limitations--operational-boundaries)
- [14. License & Author Info](#14-license--author-info)

---

## 1. Executive Summary & Industry Problem

Modern 5G cellular communication networks generate massive streams of multivariate telemetry every second across radio cells, base station schedulers, and transport backhauls. 

Traditional Network Operations Centers (NOCs) face two critical operational bottlenecks:
1. **Alert Fatigue:** Static threshold monitoring triggers floods of single-metric alarms while failing to identify subtle multi-parameter degradation patterns.
2. **Delayed Fault Diagnosis:** Diagnosing intermittent link degradation (e.g., radio frequency interference, queue buffer bloat, or spectral efficiency collapse) requires manual log collation across disparate systems, driving up **Mean Time to Resolution (MTTR)**.

### Solution Overview
The **Telecom Network Health & Intelligent Fault Prediction System** solves this by providing:
- **Automated Multivariate Telemetry Ingestion:** Continuous ingestion of physical layer, link layer, and transport indicators directly from the 32,000-record TelecomTS benchmark.
- **Unsupervised Anomaly Detection:** An automated Scikit-Learn **Isolation Forest** pipeline detecting subtle data drift without requiring labeled anomalies.
- **Multi-Model Predictive Benchmarking:** Side-by-side comparison of **Logistic Regression**, **Random Forest**, and **Gradient Boosting** classifiers with confusion matrices and feature importance rankings.
- **Transparent Composite Health Scoring:** An explainable 0–100 **Network Health Score** with customizable SLA deduction thresholds.
- **Rule-Based Root Cause Hypotheses:** Transparent symptom-to-cause diagnostic mappings coupled with practical troubleshooting steps.

---

## 2. System Architecture & Engineering Workflow

```
                                ┌──────────────────────────────────────────┐
                                │      Hugging Face: AliMaatouk/TelecomTS  │
                                │   (32,000 chunks, 33 JSONL split files)  │
                                └────────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────────┐
                                │           data_loader.py Engine          │
                                │ • Streamlit caching (@st.cache_data)     │
                                │ • Nested schema inspector & timeseries   │
                                │ • Pre-extracted fallback (3,235 samples) │
                                └────────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────────┐
                                │          preprocessing.py Engine         │
                                │ • Missing value imputation (col median)  │
                                │ • Dynamic numeric KPI column detection   │
                                │ • Standard scaling (Z-score norm)        │
                                └────────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────────┐
                                │       feature_engineering.py Module      │
                                │ • Combined Throughput (KB & Mbps)        │
                                │ • Radio Quality Index (RSRP + SNR)       │
                                │ • Max BLER & Peak PRB Congestion Index   │
                                └──────────────┬───────────────────────────┘
                                               │
                    ┌──────────────────────────┴───────────────────────────┐
                    ▼                                                      ▼
     ┌─────────────────────────────┐                        ┌─────────────────────────────┐
     │   Unsupervised ML Pipeline  │                        │   Supervised ML Pipeline    │
     │      (anomaly_detection.py) │                        │      (model_training.py)    │
     │ • Scikit-learn Isolation    │                        │ • Stratified train/test split│
     │   Forest algorithm          │                        │ • Logistic Regression        │
     │ • Contamination rate slider │                        │ • Random Forest Classifier  │
     │ • Decision score histogram  │                        │ • Gradient Boosting Clf     │
     │ • KPI drift delta analysis  │                        │ • Confusion matrix heatmaps │
     └──────────────┬──────────────┘                        └──────────────┬──────────────┘
                    │                                                      │
                    └──────────────────────────┬───────────────────────────┘
                                               │
                                               ▼
                                ┌──────────────────────────────────────────┐
                                │          network_health.py Engine        │
                                │ • Transparent 0 - 100 composite score    │
                                │ • SLA penalty deductions (BLER, PRB, etc)│
                                │ • Status: Healthy / Good / Warn / Crit   │
                                └────────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────────┐
                                │         recommendations.py Engine        │
                                │ • Rule-based fault category diagnostics  │
                                │ • Possible causes & next-step actions    │
                                └────────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                ┌──────────────────────────────────────────┐
                                │        Streamlit Interactive Web NOC     │
                                │ • 11 Interactive Navigation Pages        │
                                │ • Plotly Charts • Real-time Simulator    │
                                │ • Custom CSV Ingestion & Report Export   │
                                └──────────────────────────────────────────┘
```

---

## 3. Key Platform Features & Dashboard Modules

The web interface is structured into **11 dedicated operational pages**:

| # | Page Name | Core Functionality |
| :---: | :--- | :--- |
| **1** | **Home** | Executive NOC dashboard, KPI metrics, status pie charts, and project disclaimer. |
| **2** | **Dataset Overview** | Schema inspection, null value audit, column type explorer, and CSV export. |
| **3** | **Network KPI Dashboard** | High-level KPI aggregations (Mean, Min, Max, Median, Std), time-series trends, correlation matrices, and raw micro-waveform traces. |
| **4** | **EDA** | Bivariate scatter relationships, categorical splits (Zone, App, Mobility), and multi-metric boxplots. |
| **5** | **Anomaly Detection** | Isolation Forest pipeline with tunable contamination, score distributions, and normal vs. anomaly KPI drift table. |
| **6** | **ML Prediction** | Supervised model training comparison (Accuracy, Precision, Recall, F1, ROC-AUC) and an interactive slider-based inference simulator. |
| **7** | **Network Health** | 0–100 health scoring engine with interactive SLA penalty sliders and high-risk site prioritization. |
| **8** | **Fault Analysis** | Transparent domain rule engine mapping KPI telemetry breaches to probable root cause hypotheses. |
| **9** | **Upload KPI Data** | Upload custom user CSVs, auto-validate numeric columns, and calculate real-time health scores. |
| **10** | **Reports** | Automated executive telemetry summary with instant Markdown and CSV audit exports. |
| **11** | **About Project** | Technical architecture overview and complete 14-question interview cheatsheet. |

---

## 4. Telecom Domain Knowledge & KPI Telemetry

The platform tracks and evaluates standard 3GPP cellular performance indicators:

| Telemetry KPI | Full Name | Standard Range | Engineering Significance |
| :--- | :--- | :---: | :--- |
| **`RSRP`** | Reference Signal Received Power | `-125` to `-65 dBm` | Measures signal strength from the base station. Values below `-105 dBm` represent cell-edge conditions. |
| **`UL_SNR`** | Uplink Signal-to-Noise Ratio | `0` to `30 dB` | Compares received signal power to background noise and interference. Low SNR (< 8 dB) causes packet corruption. |
| **`DL_BLER`** | Downlink Block Error Rate | `0.0` to `1.0` | Proportion of transport blocks with uncorrectable CRC errors. Telecom target is `< 0.10` (< 10%). |
| **`UL_BLER`** | Uplink Block Error Rate | `0.0` to `1.0` | Uplink block error rate. Elevated BLER triggers heavy HARQ/ARQ retransmissions. |
| **`DL_MCS`** | Downlink Modulation & Coding Scheme | `0` to `28` | Indicates radio spectral efficiency (e.g., QPSK, 16QAM, 64QAM, 256QAM). Low MCS (< 8) represents fallback. |
| **`UL_MCS`** | Uplink Modulation & Coding Scheme | `0` to `28` | Modulation order selected by gNodeB uplink scheduler based on channel quality reports. |
| **`PRB_Utilization_DL`** | Downlink Physical Resource Block Util | `0%` to `100%` | Radio frequency bandwidth occupancy. Sustained utilization `> 85%` indicates heavy cell congestion. |
| **`Estimated_UL_Buffer`** | Uplink Buffer Backlog Size | `0` to `100,000+ B` | MAC layer buffer backlog waiting for radio grants. Direct indicator of queueing delay and latency bloat. |
| **`TX_Bytes` / `RX_Bytes`** | Transmitted / Received Bytes | `0` to `MBs` | Volume of user plane data transferred during the 10-second observation chunk. |

---

## 5. Dataset Architecture: AliMaatouk/TelecomTS

The system ingests the public **TelecomTS** benchmark dataset from Hugging Face:
- **Repository ID:** `AliMaatouk/TelecomTS`
- **Scope:** 32,000 observation windows captured across 33 chunked JSONL records.
- **Environments:** Zone A, Zone B, Zone C, and In-Motion mobility profiles.
- **Applications:** High-throughput File Download, YouTube Video Streaming, and Twitch Interactive Streaming.
- **Ground Truth Anomaly Scenarios:**
  - **Jamming:** Active external RF interference causing SNR collapse and severe BLER spikes.
  - **Buffer Overflow (Gradual & Sudden):** Uplink buffer backlog exponential growth.
  - **Co-Channel Interference (Mild & Severe):** Channel cross-talk degrading modulation orders.
  - **Antenna Hardware Failure:** Rapid attenuation of received signal power.
  - **Doppler Shift:** High-velocity mobility causing rapid phase variations.
  - **Faulty Handover Algorithms:** Excessive ping-pong handover transitions between sectors.

---

## 6. Machine Learning Pipelines

### Unsupervised Anomaly Detection (Isolation Forest)
The anomaly detection pipeline operates unsupervised using Scikit-Learn's `IsolationForest`:
1. **Feature Extraction:** Selects numeric KPIs (`RSRP`, `SNR`, `DL_BLER`, `MCS`, `PRB_Utilization`, `Buffer`).
2. **Missing Value Imputation:** Column-wise median imputation ensures resilience against missing telemetry points.
3. **Z-Score Normalization:** StandardScaler standardizes variables across differing units (dBm, ratios, bytes).
4. **Partition Tree Isolation:** Evaluates path lengths to isolate points. Anomalous records require fewer splits.
5. **Drift Analysis:** Automatically compares the mean feature values of normal vs. anomalous points to highlight the primary metrics driving deviation.

### Supervised Multi-Model Fault Classification
The system benchmarks three complementary algorithms on ground-truth network condition labels:
1. **Logistic Regression:** Linear baseline with L2 penalty, providing interpretable odds ratios.
2. **Random Forest Classifier:** Ensemble of 100 bootstrap trees, robust against outliers and non-linear interactions.
3. **Gradient Boosting Classifier:** Sequentially boosted decision trees optimizing pseudo-residuals, yielding high precision and recall on rare fault types.

#### Evaluation Metric Comparison
All models are evaluated on a **stratified test set (25%)**:
- **Accuracy:** Overall correctness across all condition classes.
- **Precision:** Minimizes false alarm dispatches.
- **Recall:** Guarantees critical anomalies and faults are captured.
- **F1-Score:** Harmonic balance of precision and recall (used as primary ranking metric).
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve for binary classifications.
- **Confusion Matrix:** Interactive Plotly heatmap revealing misclassifications.
- **Gini Feature Importance:** Highlights the most predictive telemetry features (e.g., `Estimated_UL_Buffer` and `PRB_Utilization_DL`).

---

## 7. Transparent Network Health Scoring Engine

The **Network Health Score** provides a transparent 0–100 composite index calculated through rule-based SLA penalty deductions:

$$\text{Health Score} = \max\left(0, 100 - \sum \text{Penalties}\right)$$

### Penalty Deduction Matrix
Deductions are only applied if the telemetry metric exists in the record:
- **Block Error Rate (BLER):**
  - Warning ($> 5\%$): `-15 pts`
  - Critical ($> 15\%$): `-30 pts`
- **Signal Power (RSRP):**
  - Warning ($< -95\text{ dBm}$): `-8 pts`
  - Critical ($< -110\text{ dBm}$): `-15 pts`
- **Signal Quality (SNR):**
  - Warning ($< 12\text{ dB}$): `-5 pts`
  - Critical ($< 5\text{ dB}$): `-10 pts`
- **Radio Congestion (PRB Util):**
  - Warning ($> 75\%$): `-10 pts`
  - Critical ($> 90\%$): `-20 pts`
- **Buffer Backlog (Queue Delay):**
  - Warning ($> 20\text{ KB}$): `-7 pts`
  - Critical ($> 50\text{ KB}$): `-15 pts`
- **Modulation Degradation (MCS):**
  - Warning ($< 10$): `-5 pts`
  - Critical ($< 5$): `-10 pts`

### Health Category Brackets
- 🟢 **90 – 100 | Healthy:** Optimal operational performance within SLA margins.
- 🔵 **70 – 89 | Good:** Minor radio fading or moderate load; no user impact.
- 🟡 **50 – 69 | Warning:** Elevated errors or buffer backlog; attention needed.
- 🔴 **0 – 49 | Critical:** Severe degradation (jamming, congestion); immediate NOC intervention required.

---

## 8. Expert Rule-Based Troubleshooting Engine

The diagnostic engine maps multivariate telemetry patterns directly to probable root cause hypotheses:

### Rule 1: Congestion & Buffer Overflow
- **Condition:** $\text{PRB Utilization} \ge 85\% \land \text{Buffer Backlog} \ge 25,000\text{ B}$
- **Possible Cause:** Traffic demand exceeding available radio bearer capacity, causing MAC queue backlog.
- **Recommended Action:** Review cell admission control thresholds, activate inter-frequency load balancing, and audit top-consuming subscriber sessions.

### Rule 2: RF Interference & Active Jamming
- **Condition:** $\text{SNR} \le 8\text{ dB} \land \text{DL BLER} \ge 10\% \land \text{RSRP} > -105\text{ dBm}$
- **Possible Cause:** Active co-channel interference, unauthorized transmitter, or antenna feeder noise.
- **Recommended Action:** Conduct RF spectrum sweep, audit PCI mod-3 clashes with neighbouring cells, and inspect antenna feeder line grounding.

### Rule 3: Weak Coverage & Cell Edge
- **Condition:** $\text{RSRP} \le -110\text{ dBm} \land \text{SNR} \le 5\text{ dB}$
- **Possible Cause:** Terminal operating at cell periphery, shadow fading, or excessive antenna downtilt.
- **Recommended Action:** Audit antenna mechanical and electrical downtilt, verify feeder VSWR (Voltage Standing Wave Ratio), and assess small-cell densification.

### Rule 4: Link Layer Degradation
- **Condition:** $\text{DL BLER} \ge 15\%$
- **Possible Cause:** Suboptimal Outer Loop Link Adaptation (OLLA), rapid multi-path channel fading, or transport packet discard.
- **Recommended Action:** Review CQI reporting periodicity, check HARQ retransmission limits, and inspect Ethernet backhaul link error counters.

---

## 9. Repository Directory Structure

```
Telecom Network Health & Fault Prediction System/
├── app.py                      # Core Streamlit 11-page web application
├── requirements.txt            # Python dependencies (Streamlit, Sklearn, Plotly, etc.)
├── runtime.txt                 # Pinned Python version (python-3.10) for Cloud deployment
├── README.md                   # Complete GitHub documentation & interview guide
├── .gitignore                  # Git exclusions for environments, caches, and models
│
├── src/                        # Modular Python source package
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # Hugging Face loader, caching, & schema inspector
│   ├── preprocessing.py        # Data cleaning, column detection, & scaling
│   ├── feature_engineering.py  # Domain features (Throughput, Quality indices)
│   ├── network_health.py       # Transparent 0-100 scoring engine
│   ├── anomaly_detection.py    # Scikit-learn Isolation Forest pipeline
│   ├── model_training.py       # Multi-model supervised classification & metrics
│   ├── prediction.py           # Real-time inference & risk evaluator
│   ├── recommendations.py      # Rule-based fault hypotheses & troubleshooting
│   ├── eda.py                  # Interactive Plotly charts & waveform visualizers
│   └── utils.py                # NOC styling, metric cards, & CSV export
│
├── data/
│   ├── .gitkeep
│   └── telecomts_sample.csv    # Representative 3,235-record balanced sample dataset
├── models/
│   └── .gitkeep
└── screenshots/
    └── .gitkeep
```

---

## 10. Quickstart & Local Installation

### Prerequisites
- Python **3.10** installed on your system.
- Git installed.

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/telecom-network-health.git
cd telecom-network-health
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows
py -3.10 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.10 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```
Open your browser to **`http://localhost:8501`**.

---

## 11. Streamlit Community Cloud Deployment Guide

Deploy this project online with a live shareable URL in minutes:

1. **Push Code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete telecom network health & fault prediction system"
   git branch -M main
   git remote add origin https://github.com/<your-username>/telecom-network-health.git
   git push -u origin main
   ```
   *(Note: `data/telecomts_sample.csv` is under 3MB and safe to commit. Heavy raw caches are ignored by `.gitignore`.)*

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **New app**.
   - Select your repository: `<your-username>/telecom-network-health`.
   - Set Branch to `main`.
   - Set Main file path to `app.py`.
   - Click **Deploy!**

Streamlit automatically detects `runtime.txt` (`python-3.10`) and installs all dependencies from `requirements.txt`.

---

## 12. Technical Interview Preparation Masterclass

### 30-Second Elevator Pitch:
> *"I developed a Telecom Network Health and Intelligent Fault Prediction System using Python and Streamlit. Utilizing the public 5G benchmark dataset TelecomTS, the system monitors physical and link-layer KPIs, detects multivariate anomalies using unsupervised Isolation Forests, benchmarks supervised classification models like Gradient Boosting and Random Forest, calculates an explainable 0–100 Network Health Score, and delivers rule-based troubleshooting interventions. I used Pandas for telemetry manipulation, Scikit-learn for modeling, Plotly for operations dashboards, and Streamlit for the web interface. This project gave me deep hands-on experience in proactive fault detection and cellular performance analysis."*

---

### Technical Interview Q&A (14 Questions)

#### 1. Why did you choose this project?
> *"In modern cellular networks, base stations and transport backhauls generate millions of performance counters daily. Static threshold alarms create noise or miss subtle multi-parameter degradations. Applying machine learning to telecom telemetry enables automated anomaly detection, accelerates Mean Time to Resolution (MTTR), and prevents subscriber churn before outages occur."*

#### 2. Why did you use TelecomTS over synthetic dummy data?
> *"TelecomTS is a realistic open-source 5G benchmark dataset containing actual radio measurements (RSRP, SNR, BLER, MCS, PRB utilization, Buffer delays) recorded across multiple zones, mobility states, and real anomaly scenarios like active jamming, antenna failure, and buffer overflows. Using real telemetry demonstrates genuine domain competence."*

#### 3. Why Python?
> *"Python is the undisputed industry standard for telecom data science and telemetry engineering. It offers a mature, high-performance ecosystem including Pandas and NumPy for vectorized time-series manipulation, Scikit-learn for machine learning, and Hugging Face integration."*

#### 4. Why Streamlit?
> *"Streamlit allows data scientists to build interactive, production-grade dashboards with reactive state caching, custom CSS styling, and seamless cloud deployment without the overhead of maintaining a separate frontend framework."*

#### 5. Why Isolation Forest for anomaly detection?
> *"Isolation Forest isolates anomalies by randomly partitioning feature space. Because anomalies are few and attribute-different, they have significantly shorter path lengths in isolation trees. It operates in $O(n)$ linear time, makes no Gaussian assumptions, and handles multi-metric correlations seamlessly."*

#### 6. What is Latency in a cellular network?
> *"Latency is the round-trip time required for a packet to traverse the network from user equipment through the radio access network (RAN), transport backhaul, and core network back. In our dataset, buffer backlogs (`Estimated_UL_Buffer`) serve as a primary proxy for queue delay and buffer bloat."*

#### 7. What is Throughput?
> *"Throughput is the rate of successful data delivery over a communication channel, measured in bits or bytes per second. In our platform, it is derived from `TX_Bytes` and `RX_Bytes` over the 10-second sampling window."*

#### 8. What is Packet Loss?
> *"Packet loss is the percentage of transmitted data packets that fail to reach their destination. In cellular networks, this is directly monitored via Downlink and Uplink Block Error Rate (`DL_BLER` and `UL_BLER`)."*

#### 9. What is Network Congestion?
> *"Network congestion occurs when traffic demand exceeds available carrier carrying capacity, characterized by high PRB utilization ($> 85\%$), expanding MAC buffer backlogs, and elevated block retransmissions."*

#### 10. What is a KPI?
> *"A Key Performance Indicator (KPI) is a quantifiable metric used to measure network operational performance, link quality, and SLA compliance (e.g., RSRP, BLER, PRB Utilization, Throughput)."*

#### 11. What is Anomaly Detection?
> *"Anomaly detection is the identification of rare events or telemetry observations that deviate significantly from baseline operational patterns, signaling hardware faults, external interference, or configuration bugs."*

#### 12. How does your Health Score work?
> *"The score starts at 100 points and applies transparent, domain-defined deductions whenever KPIs breach configurable thresholds: BLER (up to -30), Signal Quality (up to -25), Congestion (up to -20), Buffer Backlog (up to -15), and Modulation (up to -10)."*

#### 13. How does this help a telecom company?
> *"It cuts Mean Time to Detect (MTTD) and Mean Time to Repair (MTTR), prevents customer churn from undetected service degradation, and provides field technicians with symptom-to-cause hypotheses for faster troubleshooting."*

#### 14. What are the project limitations?
> *"As an analytical software prototype evaluated on public benchmark data, it does not send write commands to live cell towers; operational deployment would require integration with OSS/EMS mediation platforms."*

---

## 13. Engineering Limitations & Operational Boundaries
1. **Prototype Boundary:** Evaluated on benchmark data; does not send direct MML (Man-Machine Language) or NETCONF/YANG control commands to live gNodeB base stations.
2. **Probabilistic Hypotheses:** Fault diagnoses represent evidence-based hypotheses, not definitive physical hardware determinations.
3. **Zero Paid APIs:** The entire pipeline relies exclusively on open-source Python libraries and local Hugging Face caching.

---

## 14. License & Author Info
- **License:** MIT License. Free for academic, personal, and interview demonstration use.
- **Dataset Attribution:** Ali Maatouk et al., *TelecomTS: A Time-Series Benchmark Dataset for 5G Telemetry*.
