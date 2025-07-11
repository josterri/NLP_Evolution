import streamlit as st

# Import chapter controllers
from chapter1 import render_chapter_1
from chapter2 import render_chapter_2
from chapter3 import render_chapter_3
from chapter4 import render_chapter_4
from chapter5 import render_chapter_5
from chapter6 import render_chapter_6
from chapter7 import render_chapter_7
from chapter8 import render_chapter_8

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

    # A dictionary for the new 8-chapter structure
    CHAPTERS = {
        "Chapter 1: The Statistical Era": render_chapter_1,
        "Chapter 2: The Rise of Neural Networks & Embeddings": render_chapter_2,
        "Chapter 3: Sequential Models & The Power of Context": render_chapter_3,
        "Chapter 4: The Transformer Revolution": render_chapter_4,
        "Chapter 5: Applying the Foundations: Text Classification": render_chapter_5,
        "Chapter 6: The Rise of Generative Models": render_chapter_6,
        "Chapter 7: Applying Generative Models: Text Summarization": render_chapter_7,
        "Chapter 8: The Era of Large Language Models (LLMs)": render_chapter_8,
    }

    selection = st.sidebar.radio("Go to Chapter:", list(CHAPTERS.keys()))

    st.title("📜 The Evolution of NLP")
    st.markdown("From Simple Predictions to Deep Understanding")
    st.markdown("---")

    # Call the render function for the selected chapter
    # Each chapter's render function is now responsible for its own sub-navigation
    page = CHAPTERS[selection]
    page()

    st.sidebar.markdown("---")
    st.sidebar.info("""
    This app demonstrates the key milestones in the evolution of Natural Language Processing,
    from simple statistical models to the foundations of modern AI.
    """)

if __name__ == "__main__":
    main()
