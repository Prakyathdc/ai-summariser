"""
UI Styles Module
================
Injects custom CSS for a beautiful Streamlit UI.
"""
import streamlit as st

def apply_custom_styles():
    """Inject custom CSS for styling."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .main-title {
            color: #1e3a8a;
            font-family: 'Inter', sans-serif;
            text-align: center;
        }
        .summary-box {
            background-color: #ffffff;
            border-left: 5px solid #3b82f6;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
