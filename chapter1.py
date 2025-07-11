import streamlit as st

# Import the render functions from the new sub-chapter files
from chapter1_1 import render_1_1
from chapter1_2 import render_1_2
from chapter1_3 import render_1_3

def render_chapter_1():
    """Renders all content for Chapter 1 by controlling sub-chapters."""
    st.header("Chapter 1: The Foundation - Predicting the Next Word")

    SUB_CHAPTERS = {
        "1.1: N-grams & The Interactive Demo": render_1_1,
        "1.2: The Sparsity Problem": render_1_2,
        "1.3: Smoothing Techniques": render_1_3,
    }

    sub_selection = st.sidebar.radio(
        "Chapter 1 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch1_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
