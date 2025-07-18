import streamlit as st

# Import the render functions from the sub-chapter files
from chapter3_1 import render_3_1
from chapter3_2 import render_3_2
from chapter3_3 import render_3_3
from chapter3_4 import render_3_4
from chapter3_5 import render_section_3_5

# Import interactive modules
from code_exercises import render_exercise_widget

def render_chapter_3():
    """Renders all content for Chapter 3 by controlling sub-chapters."""
    st.header("Chapter 3: Sequential Models & The Power of Context")

    SUB_CHAPTERS = {
        "3.1: The Limits of a Dictionary (Polysemy)": render_3_1,
        "3.2: The Idea of a 'Rolling Context'": render_3_2,
        "3.3: The Breakthrough - Words in Disguise": render_3_3,
        "3.4: The Attention Mechanism - Focusing on What Matters": render_3_4,
        "3.5: RNNs and LSTMs - Sequential Neural Networks": render_section_3_5,
    }

    sub_selection = st.sidebar.radio(
        "Chapter 3 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch3_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
    
    # Add coding practice at the end of the chapter
    st.markdown("---")
    render_exercise_widget("chapter3")
