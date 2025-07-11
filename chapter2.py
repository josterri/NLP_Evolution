import streamlit as st

# Import the render functions from the sub-chapter files
from chapter2_1 import render_2_1
from chapter2_2 import render_2_2
from chapter2_3 import render_2_3

def render_chapter_2():
    """Renders all content for Chapter 2 by controlling sub-chapters."""
    st.header("Chapter 2: The Rise of Neural Networks & Embeddings")

    SUB_CHAPTERS = {
        "2.1: The Concept of Word Embeddings": render_2_1,
        "2.2: Exploring the Vector Space & Analogies": render_2_2,
        "2.3: Limitations of Static Embeddings": render_2_3,
    }

    sub_selection = st.sidebar.radio(
        "Chapter 2 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch2_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
