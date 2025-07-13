import streamlit as st

# Import the render functions from the sub-chapter files
from chapter8_0 import render_8_0
from chapter8_1 import render_8_1
from chapter8_2 import render_8_2

def render_chapter_8():
    """Renders all content for Chapter 8 by controlling sub-chapters."""
    st.header("Chapter 8: The Era of Large Language Models (LLMs)")

    SUB_CHAPTERS = {
        "8.0: The Cambrian Explosion of AI": render_8_0,
        "8.1: The Modern Landscape": render_8_1,
        "8.2: The Future & Frontiers of NLP": render_8_2,
    }

    # Use a selectbox for sub-navigation
    sub_selection = st.selectbox(
        "Chapter 8 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch8_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
