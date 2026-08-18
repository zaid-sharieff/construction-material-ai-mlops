import streamlit as st
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from frontend.ui.detect_quantify import render as detect_page
from frontend.ui.dashboard import render as dashboard_page
from frontend.ui.grafana_monitoring import render as grafana_page
from frontend.ui.drift_report import render as drift_page
from frontend.ui.chatbot import render as chatbot_page

st.markdown(
    """
    <style>
    /* Hide default Streamlit multipage navigation */
    section[data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Construction Material AI",
    page_icon="🏗️",
    layout="wide"
)

# ---------------- GLOBAL STYLES ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top, #0f2027, #000000);
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #00E5FF;
    }

    .sidebar-sub {
        font-size: 14px;
        color: #9CA3AF;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🏗️ Material AI</div>", unsafe_allow_html=True)
    #st.markdown("<div class='sidebar-sub'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📸 Detect & Quantify", "📈 Analytics Dashboard", "📊 Grafana Monitoring", "📉 Drift Analysis", "🤖 Chatbot"],
        index=0,
        label_visibility="collapsed"
    )

# ---------------- ROUTING ----------------
if page == "🏠 Home":
    st.title("🏗️ Construction Material Classification & Quantification")

    st.markdown(
        """
        This application uses **AI-based Computer Vision & LLMs** to:

        - Identify construction materials from images  
        - Detect material regions  
        - Count discrete materials  
        - Estimate volume & mass for granular materials  
        - Provide intelligent explanations using AI  

        ⬅️ Use the sidebar to navigate.
        """
    )

    st.info("This system is designed for educational and analytical purposes.")

elif page == "📸 Detect & Quantify":
    detect_page()

elif page == "📈 Analytics Dashboard":
    dashboard_page()

elif page == "📉 Drift Analysis":
    drift_page()

elif page == "📊 Grafana Monitoring":
    grafana_page()

elif page == "🤖 Chatbot":
    chatbot_page()
