"""
History Panel UI
==================
Displays past summaries saved in the SQLite database.
"""
import streamlit as st
import pandas as pd

def render_history_panel(db_manager):
    """Render the database history in a Streamlit dataframe."""
    st.header("🕰️ Summary History")
    
    records = db_manager.get_all_summaries()
    
    if not records:
        st.info("No summaries generated yet. Try summarizing some text first!")
        return

    # Convert to DataFrame for nice display
    df = pd.DataFrame(records)
    
    # Select columns to display
    display_df = df[['created_at', 'model_name', 'original_word_count', 'compression_ratio', 'summary_text']]
    display_df.columns = ['Date', 'Model', 'Words', 'Compression', 'Summary Snippet']
    
    # Truncate summary for the table view
    display_df['Summary Snippet'] = display_df['Summary Snippet'].apply(
        lambda x: x[:100] + "..." if len(x) > 100 else x
    )

    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("Search Past Summaries")
    search_term = st.text_input("Search by keyword:")
    
    if search_term:
        filtered = df[df['original_text'].str.contains(search_term, case=False, na=False) | 
                      df['summary_text'].str.contains(search_term, case=False, na=False)]
        st.write(f"Found {len(filtered)} results:")
        for _, row in filtered.iterrows():
            with st.expander(f"{row['created_at']} - {row['model_name']}"):
                st.write("**Summary:**", row['summary_text'])
                st.write("**Original Snippet:**", row['original_text'][:300] + "...")
