import streamlit as st

# Import the render functions from the sub-chapter files
from chapter7_0 import render_7_0
from chapter7_1 import render_7_1
from chapter7_2 import render_7_2
from chapter7_3 import render_7_3
from chapter7_4 import render_7_4
from chapter7_5 import render_7_5
from chapter7_6 import render_7_6

def render_chapter_7():
    """Renders all content for Chapter 7 by controlling sub-chapters."""
    st.header("Chapter 7: Build Your Own Generative Model")

    SUB_CHAPTERS = {
        "7.0: The Roadmap - Building a 'nano-GPT'": render_7_0,
        "7.1: Step 1 - The Data and the Tokenizer": render_7_1,
        "7.2: Step 2 - Building the Model Components": render_7_2,
        "7.3: Step 3 - Assembling the Transformer Block": render_7_3,
        "7.4: Step 4 - Creating the Full Model": render_7_4,
        "7.5: Step 5 - The Training Loop": render_7_5,
        "7.6: The Grand Finale - Full Code & Generation": render_7_6,
    }

    # Use a selectbox for sub-navigation
    sub_selection = st.selectbox(
        "Chapter 7 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch7_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
