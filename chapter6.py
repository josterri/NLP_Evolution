import streamlit as st

# Import the render functions from the sub-chapter files
from chapter6_0 import render_6_0
from chapter6_1 import render_6_1
from chapter6_2 import render_6_2
from chapter6_3 import render_6_3
from chapter6_4 import render_6_4

def render_chapter_6():
    """Renders all content for Chapter 6 by controlling sub-chapters."""
    st.header("Chapter 6: The Rise of Generative Models")

    SUB_CHAPTERS = {
        "6.0: The Story So Far - Next-Word Prediction": render_6_0,
        "6.1: The Decoder-Only Architecture (GPT-style)": render_6_1,
        "6.2: The Training Objective: Causal Language Modeling": render_6_2,
        "6.3: The Emergence of In-Context Learning": render_6_3,
        "6.4: Interactive Generation Workbench": render_6_4,
    }

    # Use a selectbox for sub-navigation
    sub_selection = st.selectbox(
        "Chapter 6 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch6_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
