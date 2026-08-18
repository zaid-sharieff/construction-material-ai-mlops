import streamlit as st
from pathlib import Path
import os
import sys

# Fix import path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.drift.drift_report import generate_drift_report

def render():
    st.markdown(
        "<h1 style='color:#00E5FF;'>📉 Data Drift Analysis</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This section analyzes **prediction confidence drift** using **Evidently AI**.

        **Reference window:** Older predictions  
        **Current window:** Latest predictions  
        **Metric:** Confidence Value Drift
        """
    )

    report_path = Path("drift_report/drift_report.html")

    # Session state to track execution
    if "drift_ran" not in st.session_state:
        st.session_state.drift_ran = False

    # ----------------------------
    # Run Drift Button
    # ----------------------------
    if st.button("🚨 Run Drift Analysis"):
        with st.spinner("Running drift detection..."):
            success, message = generate_drift_report()

        if success:
            st.session_state.drift_ran = True
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")

    # ----------------------------
    # Render Drift Report
    # ----------------------------
    if st.session_state.drift_ran and report_path.exists():
        st.subheader("📊 Evidently Drift Report")

        html = report_path.read_text(encoding="utf-8")

        st.components.v1.html(
            html,
            height=900,
            scrolling=True
        )
    else:
        st.info("Run drift analysis to generate and view the report.")
