# frontend/pages/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sqlalchemy.orm import Session
from sqlalchemy import func
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.database.db import SessionLocal
from src.database.models import Image, Detection, Quantification


def render():

    # ================== NEON THEME ==================
    pio.templates["neon_dark"] = pio.templates["plotly_dark"]
    pio.templates["neon_dark"].layout.update(
        font=dict(color="#E6EDF3"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        colorway=[
            "#00E5FF", "#FF4DFF", "#00FF9C",
            "#FFD166", "#FF6B6B", "#9D4EDD"
        ]
    )
    pio.templates.default = "neon_dark"

    # ================== HEADER ==================
    st.markdown(
        "<h1 style='color:#00E5FF; text-shadow:0 0 12px rgba(0,229,255,0.45)'>📈 Material Analytics Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Advanced insights from **AI-based material analysis pipelines**")

    db: Session = SessionLocal()

    # ================== KPI METRICS ==================
    c1, c2, c3 = st.columns(3)
    c1.metric("📷 Images", db.query(func.count(Image.id)).scalar())
    c2.metric("🔍 Detections", db.query(func.count(Detection.id)).scalar())
    c3.metric("📦 Quantifications", db.query(func.count(Quantification.id)).scalar())

    st.markdown("---")

    # ================== MATERIAL DISTRIBUTION ==================
    material_df = pd.DataFrame(
        db.query(
            Quantification.material,
            func.count(Quantification.id).label("count")
        ).group_by(Quantification.material).all(),
        columns=["material", "count"]
    )

    if not material_df.empty:
        col1, col2 = st.columns(2)

        # BAR – absolute frequency
        col1.plotly_chart(
            px.bar(
                material_df,
                x="material",
                y="count",
                color="material",
                title="📌 Material Frequency"
            ).update_layout(title_x=0.5),
            width='stretch'
        )

        # DONUT – proportional share
        col2.plotly_chart(
            px.pie(
                material_df,
                names="material",
                values="count",
                hole=0.55,
                title="🧠 Material Share"
            ).update_traces(textinfo="percent+label"),
            width='stretch'
        )

    st.markdown("---")

    # ================== DETECTION DISTRIBUTION ==================
    det_df = pd.DataFrame(
        db.query(
            Detection.material,
            func.count(Detection.id).label("detections")
        ).group_by(Detection.material).all(),
        columns=["material", "detections"]
    )

    if not det_df.empty:
        st.plotly_chart(
            px.box(
                det_df,
                x="material",
                y="detections",
                color="material",
                title="📦 Detection Distribution (Spread Analysis)"
            ).update_layout(title_x=0.5),
            width='stretch'
        )

    st.markdown("---")

    # ================== SAND & SOIL QUANTIFICATION ==================
    sand_df = pd.DataFrame(
        db.query(
            Quantification.material,
            Quantification.volume_m3,
            Quantification.mass_kg
        )
        .filter(Quantification.material.in_(["sand", "soil"]))
        .all(),
        columns=["material", "volume", "mass"]
    )

    if not sand_df.empty:
        col1, col2 = st.columns(2)

        # GROUPED BAR – volume comparison
        col1.plotly_chart(
            px.bar(
                sand_df,
                x="material",
                y="volume",
                color="material",
                title="🌊 Volume Comparison (m³)"
            ).update_layout(title_x=0.5),
            width='stretch'
        )

        # SCATTER – volume vs mass relationship
        col2.plotly_chart(
            px.scatter(
                sand_df,
                x="volume",
                y="mass",
                color="material",
                trendline="ols",
                title="⚖️ Volume vs Mass Correlation"
            ).update_layout(title_x=0.5),
            width='stretch'
        )

    st.markdown("---")

    # ================== TEMPORAL TREND ==================
    time_df = pd.DataFrame(
        db.query(
            Image.upload_time,
            Quantification.material
        )
        .join(Quantification, Image.id == Quantification.image_id)
        .all(),
        columns=["time", "material"]
    )

    if not time_df.empty:
        time_df["count"] = 1
        st.plotly_chart(
            px.line(
                time_df,
                x="time",
                y="count",
                color="material",
                markers=True,
                title="📈 Prediction Trend Over Time"
            ).update_layout(title_x=0.5),
            width='stretch'
        )

    st.markdown("---")

    # ================== RECENT PREDICTIONS ==================
    recent_df = pd.DataFrame(
        db.query(
            Image.filename,
            Image.upload_time,
            Quantification.material,
            Quantification.count,
            Quantification.volume_m3,
            Quantification.mass_kg
        )
        .join(Quantification, Image.id == Quantification.image_id)
        .order_by(Image.upload_time.desc())
        .limit(10)
        .all(),
        columns=["filename", "time", "material", "count", "volume", "mass"]
    )

    st.subheader("🕒 Recent Predictions")
    if not recent_df.empty:
        st.dataframe(recent_df, width='stretch', hide_index=True)
    else:
        st.info("No predictions logged yet.")
