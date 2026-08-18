# frontend/ui/grafana_monitoring.py

import streamlit as st

def render():
    st.markdown(
        "<h1 style='color:#00E5FF; text-shadow:0 0 12px rgba(0,229,255,0.4)'>📊 Grafana Monitoring</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This dashboard provides **system-level and pipeline monitoring**
        for the Construction Material AI project using Grafana.
        """
    )

    grafana_url = (
        "http://localhost:3000/d/"
        "e31e5e37-1489-43c7-9a4a-09d30906fac9/"
        "construction-material-ai-e28093-monitoring"
        "?orgId=1"
        "&kiosk=tv"
        "&refresh=10s"
    )

    st.components.v1.iframe(
        src=grafana_url,
        height=900,
        scrolling=True
    )

    st.info(
        "If the dashboard does not load, ensure Grafana is running "
        "and embedding is enabled."
    )
