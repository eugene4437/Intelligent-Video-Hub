"""
Intelligent Video Processing and Semantic Structuring System
Main Streamlit Application Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="Intelligent Video Processing System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui.main_page import render_main_page

render_main_page()
