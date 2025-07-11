import streamlit as st
from chapter1 import render_chapter_1
from chapter3 import render_chapter_3
from chapter2 import render_chapter_2

# --- App Configuration ---
st.set_page_config(
    page_title="The Evolution of NLP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main App Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.sidebar.title("Navigation")
    
    # A dictionary mapping chapter titles to their render functions, imported from other files
    PAGES = {
        "Chapter 1: N-grams": render_chapter_1,
        "Chapter 2: Static Embeddings": render_chapter_2,
        "Chapter 3: Contextual Embeddings": render_chapter_3,
    }
    
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    st.title("📜 The Evolution of NLP")
    st.markdown("From Simple Predictions to Deep Understanding")
    st.markdown("---")

    # Call the render function for the selected page
    page = PAGES[selection]
    page()

    st.sidebar.markdown("---")
    st.sidebar.info("""
    This app demonstrates the key milestones in the evolution of Natural Language Processing, 
    from simple statistical models to the foundations of modern AI.
    """)

if __name__ == "__main__":
    main()
