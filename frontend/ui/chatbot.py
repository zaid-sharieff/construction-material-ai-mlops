# frontend/pages/chatbot.py

import streamlit as st
import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.chatbot.chatbot_engine import get_gemini_material_response


def render():
    # ------------------ HEADER ------------------
    st.markdown(
        "<h1 style='color:#00E5FF; text-shadow:0 0 12px rgba(0,229,255,0.4)'>🤖 Material AI Chatbot</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Ask questions related to:
        - Types of construction materials  
        - Properties of sand, brick, soil, wood  
        - Uses of materials in construction  
        - Density and strength characteristics  
        """
    )

    # ------------------ CHAT STATE ------------------
    if "material_chat_history" not in st.session_state:
        st.session_state.material_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am an AI assistant specialized in construction materials. "
                    "Ask me about material types, properties, or construction uses."
                )
            }
        ]

    # ------------------ DISPLAY HISTORY ------------------
    for msg in st.session_state.material_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ------------------ INPUT ------------------
    if prompt := st.chat_input("Ask a question about construction materials..."):

        # User message
        st.session_state.material_chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("AI is generating response..."):
                response = get_gemini_material_response(prompt)
                time.sleep(0.5)
            st.markdown(response)

        st.session_state.material_chat_history.append(
            {"role": "assistant", "content": response}
        )
