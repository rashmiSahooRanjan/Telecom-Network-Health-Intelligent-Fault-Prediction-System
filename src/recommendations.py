"""
Telecom Network Health & Intelligent Fault Prediction System
Recommendations & Fault Analysis Module: Rule-based fault categorizer and actionable troubleshooting guidance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def analyze_faults_and_causes(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Evaluate telecom domain expert rules across available KPIs to detect probable fault categories.
    
    Returns a list of structured fault diagnostics:
      [
        {
          "fault_category": "Network Congestion",
          "severity": "Critical" | "Warning",
          "evidence": "PRB_Utilization_DL at 92.4% (>85%) and Estimated_UL_Buffer at 45,200 B",
          "possible_cause": "Excessive user session load and traffic demand exhausting scheduled PRBs.",
          "recommended_action": "Check cell scheduler policies, review admission control thresholds, and expand carrier bandwidth or offload to small cells."
        },
        ...
      ]
    """
    detected_faults = []

    # 1. Network Congestion
    prb_dl = row.get("PRB_Utilization_DL")
    buffer_ul = row.get("Estimated_UL_Buffer")
    bler_dl = row.get("DL_BLER")
    rsrp = row.get("RSRP")
    snr = row.get("UL_SNR")
    tx_bytes = row.get("TX_Bytes")
    mcs_dl = row.get("DL_MCS")

    is_high_prb = prb_dl is not None and prb_dl >= 85.0
    is_high_buffer = buffer_ul is not None and buffer_ul >= 25000.0

    if is_high_prb and is_high_buffer:
        detected_faults.append({
            "category": "Network Congestion & Buffer Overflow",
            "severity": "Critical" if (prb_dl >= 90.0 or buffer_ul >= 45000.0) else "Warning",
            "evidence": f"Downlink PRB Utilization is {prb_dl:.1f}% (threshold: 85%) and Uplink Buffer Backlog is {buffer_ul:,.0f} B.",
            "possible_cause": "Traffic demand exceeding current radio bearer capacity, leading to queue buildup in the gNodeB / UE MAC buffers.",
            "recommended_action": "Inspect cell traffic load distribution, activate load balancing with adjacent sectors, and verify QoS scheduling priority queues."
        })
    elif is_high_prb:
        detected_faults.append({
            "category": "High Radio Resource (PRB) Utilization",
            "severity": "Warning",
            "evidence": f"Downlink PRB Utilization reached {prb_dl:.1f}% (>85%).",
            "possible_cause": "High concurrent user activity or heavy data streaming exhausting available Physical Resource Blocks.",
            "recommended_action": "Audit top talkers, assess cell capacity limits, and review carrier aggregation configurations."
        })

    # 2. Radio Frequency (RF) Interference / Jamming / Signal Degradation
    is_low_rsrp = rsrp is not None and rsrp <= -105.0
    is_low_snr = snr is not None and snr <= 8.0
    is_high_bler = bler_dl is not None and bler_dl >= 0.10

    if is_low_snr and is_high_bler and not is_low_rsrp:
        detected_faults.append({
            "category": "RF Interference / External Jamming",
            "severity": "Critical",
            "evidence": f"UL SNR collapsed to {snr:.1f} dB while RSRP remained moderate ({rsrp:.1f} dBm), accompanied by high DL BLER ({bler_dl*100:.1f}%).",
            "possible_cause": "Active external jamming, co-channel interference from misconfigured neighbouring gNodeB, or faulty RF amplifier.",
            "recommended_action": "Perform spectrum sweep for unauthorized radio transmitters, check PCI (Physical Cell ID) mod3 clashes, and inspect RF antenna filters."
        })
    elif is_low_rsrp and is_low_snr:
        detected_faults.append({
            "category": "Weak Signal Coverage / Cell Edge",
            "severity": "Warning" if rsrp > -115.0 else "Critical",
            "evidence": f"RSRP is poor at {rsrp:.1f} dBm and SNR is {snr:.1f} dB.",
            "possible_cause": "Terminal is operating at extreme cell edge, in deep indoor shadow fade, or antenna feeder line loss.",
            "recommended_action": "Verify antenna mechanical and electrical down-tilt, inspect feeder VSWR (Voltage Standing Wave Ratio), and evaluate small-cell densification."
        })

    # 3. Transmission / Link Layer Packet Loss
    if is_high_bler and not (is_low_snr and not is_low_rsrp):
        detected_faults.append({
            "category": "High Block Error Rate (BLER) / Packet Loss",
            "severity": "Critical" if bler_dl >= 0.20 else "Warning",
            "evidence": f"Downlink BLER is {bler_dl*100:.1f}% (standard telecom SLA target is < 10%).",
            "possible_cause": "Suboptimal link adaptation (inaccurate CQI reporting), rapid radio channel fading, or transmission packet drop.",
            "recommended_action": "Review Link Adaptation OLLA (Outer Loop Link Adaptation) parameters, check HARQ retransmission limits, and inspect Ethernet backhaul error counters."
        })

    # 4. Modulation Drop / Spectral Efficiency Collapse
    if mcs_dl is not None and mcs_dl <= 6.0 and not is_low_rsrp:
        detected_faults.append({
            "category": "Downlink Modulation Degradation (Low MCS)",
            "severity": "Warning",
            "evidence": f"Downlink MCS degraded to index {mcs_dl:.1f} despite passable signal power.",
            "possible_cause": "Interference bursts causing UE to report low Channel Quality Indicator (CQI), forcing fallback to QPSK / low-order modulation.",
            "recommended_action": "Analyze CQI and Rank Indicator (RI) distributions and check for intermittent multi-path reflections or Doppler effects."
        })

    # 5. Fallback: If no specific rule triggered but marked anomalous in dataset
    if not detected_faults:
        anom_flag = row.get("anomaly_present")
        anom_type = row.get("anomaly_type")
        if (str(anom_flag).lower() in ["yes", "1", "true"]) or (anom_type and str(anom_type).lower() != "normal"):
            ticket = row.get("troubleshooting_summary", "")
            detected_faults.append({
                "category": f"Network Anomaly: {anom_type}" if anom_type else "General Network Anomaly",
                "severity": "Warning",
                "evidence": f"Anomaly tagged in dataset: {anom_type or 'General deviation'}.",
                "possible_cause": "Unusual multivariate KPI correlation deviation detected by telemetry monitoring.",
                "recommended_action": ticket if ticket else "Inspect raw telemetry logs around the event timestamp and verify gNodeB alarms."
            })

    return detected_faults


def generate_recommendations(row: pd.Series) -> List[Dict[str, str]]:
    """
    Generate clean troubleshooting action items based on available KPI anomalies.
    Returns clear suggestions, explicitly stated as recommendations rather than automated commands.
    """
    faults = analyze_faults_and_causes(row)
    if not faults:
        return [{
            "category": "System Nominal",
            "severity": "Healthy",
            "status": "All monitored KPIs operate within safe engineering thresholds.",
            "action": "Continue regular automated telemetry polling and routine health checks."
        }]

    return faults
