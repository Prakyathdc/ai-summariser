"""
Analytics Panel UI
==================
Displays text statistics and charts in the Streamlit app.
"""
import streamlit as st
from utils.chart_generator import ChartGenerator

def render_analytics_panel(text: str, words: list):
    """Render charts for the input text."""
    st.subheader("📊 Visual Analytics")
    
    if len(words) < 10:
        st.warning("Not enough text to generate analytics.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Word Cloud**")
        try:
            wc_img = ChartGenerator.generate_wordcloud(text)
            st.image(wc_img, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate word cloud: {e}")
            
    with col2:
        st.markdown("**Top Words Frequency**")
        try:
            bar_img = ChartGenerator.generate_word_freq_chart(words, top_n=10)
            st.image(bar_img, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate bar chart: {e}")
