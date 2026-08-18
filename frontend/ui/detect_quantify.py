# frontend/ui/detect_quantify.py

import streamlit as st
import tempfile
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.ml.predict import predict

def render():
    st.markdown("## 📸 Detect & Quantify Construction Materials")

    uploaded_file = st.file_uploader(
        "Upload an image of construction material",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        st.image(image_path, caption="Uploaded Image", width='stretch')

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                result = predict(image_path)

            st.success("Analysis completed")

            st.markdown(f"### Material: `{result['material'].upper()}`")

            if result["count"] is not None:
                st.metric("Detected Count", result["count"])

            if result["quantity"] is not None:
                col1, col2 = st.columns(2)
                col1.metric("Volume (m³)", result["quantity"]["estimated_volume_m3"])
                col2.metric("Mass (kg)", result["quantity"]["estimated_mass_kg"])

            if result["annotated_image"] and os.path.exists(result["annotated_image"]):
                st.markdown("### 🖼️ Annotated Output")
                st.image(result["annotated_image"], width='stretch')