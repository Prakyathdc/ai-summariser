"""
Main Application Entry Point
============================
The main Streamlit script.
"""
import streamlit as st
import logging
from config import setup_logging, APP_TITLE, APP_ICON
from ui.styles import apply_custom_styles
from models.summarizer import TextSummarizer
from models.model_registry import list_models
from database.db_manager import DatabaseManager
from utils.file_parser import extract_text_from_file

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize singletons in session state
if "summarizer" not in st.session_state:
    st.session_state.summarizer = TextSummarizer()
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

apply_custom_styles()

st.markdown(f"<h1 class='main-title'>{APP_ICON} {APP_TITLE}</h1>", unsafe_allow_html=True)
st.write("A professional text summarization system powered by Hugging Face Transformers.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
model_options = {m.display_name: m.name for m in list_models()}
selected_model_display = st.sidebar.selectbox("Select Model", list(model_options.keys()))
summary_length = st.sidebar.select_slider("Summary Length", options=["Short", "Medium", "Long"], value="Medium")

# Main Content
tab1, tab2 = st.tabs(["Summarize Text", "Upload File"])

input_text = ""

with tab1:
    input_text = st.text_area("Paste your text here (min 10 words):", height=200)

with tab2:
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        try:
            bytes_data = uploaded_file.read()
            input_text = extract_text_from_file(bytes_data, uploaded_file.name)
            st.success("File parsed successfully!")
            with st.expander("Show extracted text"):
                st.write(input_text)
        except Exception as e:
            st.error(f"Error parsing file: {e}")

if st.button("Generate Summary", type="primary"):
    if not input_text or len(input_text.split()) < 10:
        st.warning("Please provide at least 10 words of text to summarize.")
    else:
        selected_model_name = model_options[selected_model_display]
        
        with st.spinner(f"Loading {selected_model_name} and generating summary..."):
            try:
                # Load model
                st.session_state.summarizer.load_model(selected_model_name)
                
                # Generate summary
                result = st.session_state.summarizer.summarize(input_text, summary_length=summary_length)
                
                # Display Summary
                st.subheader("📝 Summary")
                st.markdown(f"<div class='summary-box'>{result.summary}</div>", unsafe_allow_html=True)
                
                # Display Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Words", result.original_word_count)
                col2.metric("Summary Words", result.summary_word_count)
                col3.metric("Compression Ratio", f"{result.compression_ratio}x")
                
                st.caption(f"Generated in {result.generation_time_seconds}s using {result.model_name}")
                
                # Save to database
                st.session_state.db_manager.save_summary(
                    input_text, result.summary, result.model_name,
                    result.original_word_count, result.summary_word_count,
                    result.compression_ratio
                )
            except Exception as e:
                logger.error("Summarization error: %s", e)
                st.error(f"An error occurred during summarization: {e}")
