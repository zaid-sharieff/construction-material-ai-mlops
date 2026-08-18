import streamlit as st
from google import genai
from google.genai import types

def get_gemini_material_response(prompt: str) -> str:
    """
    Gemini-powered chatbot for construction material knowledge
    """

    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ GEMINI_API_KEY not found in Streamlit secrets."

    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

        system_instruction = (
            "You are a professional, educational AI assistant specialized ONLY in the domain of "
            "**construction materials and civil engineering fundamentals**.\n\n"

            "Your knowledge domain includes:\n"
            "- Construction materials such as brick, sand, soil, wood, concrete, and steel\n"
            "- Types and classifications of these materials\n"
            "- Engineering properties (density, strength, durability)\n"
            "- Construction uses and applications\n"
            "- Material handling and safety considerations\n\n"

            "When a question is relevant, you MUST:\n"
            "- Provide a structured explanation\n"
            "- Use bullet points\n"
            "- Clearly explain material types and properties\n"
            "- Maintain an academic and professional tone\n\n"

            "**REFUSAL RULE:**\n"
            "You MUST refuse to answer any off-topic, personal, emotional, or unrelated question.\n"
            "For such questions, respond ONLY with:\n"
            "'I am an AI assistant specialized in construction materials and can only answer "
            "questions related to construction materials, their types, properties, and uses.'\n\n"

            "You MUST conclude EVERY valid response with:\n"
            "'**Disclaimer:** The information provided is for educational and estimation purposes only "
            "and should not be considered a substitute for professional engineering judgment or on-site assessment.'"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4
            )
        )

        return response.text

    except Exception as e:
        error_msg = str(e)

        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            return (
                "⚠️ **Service Temporarily Unavailable**\n\n"
                "The AI model is currently experiencing high traffic."
                "Please wait a few seconds and try again.\n\n"
                "**Disclaimer:** Responses are for educational purposes only and should not be "
                "considered a substitute for professional engineering judgment or on-site assessment."
            )
        
        return f"❌ An API error occurred: {e}"
