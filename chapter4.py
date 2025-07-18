import streamlit as st

# Import the render functions from the sub-chapter files
from chapter4_1 import render_4_1
from chapter4_2 import render_4_2
from chapter4_3 import render_4_3
from chapter4_4 import render_4_4
from chapter4_5 import render_4_5
from chapter4_6 import render_4_6
from chapter4_7 import render_4_7
from chapter4_8 import render_4_8
from chapter4_9 import render_4_9
from chapter4_10 import render_4_10
from chapter4_11 import render_section_4_11

# Import interactive modules removed - focusing on core content

def render_chapter_4():
    """Renders all content for Chapter 4 by controlling sub-chapters."""
    st.header("Chapter 4: The Attention Mechanism")

    SUB_CHAPTERS = {
        "4.1: The Motivation for Attention": render_4_1,
        "4.2: Query, Key, and Value": render_4_2,
        "4.3: Query, Key, and Value in Detail": render_4_3,
        "4.4: The Attention Calculation Step-by-Step": render_4_4,
        "4.5: Interactive Attention Workbench": render_4_5,
        "4.6: The 'Why' Behind the Math": render_4_6,
        "4.7: Exercise - Predicting the Next 5 Words": render_4_7,
        "4.8: Python Code for Prediction Methods": render_4_8,
        "4.9: Attention Prediction - A Detailed Walkthrough": render_4_9,
        "4.10: Long Exercise - Create Your Own Predicted Text": render_4_10,
        "4.11: BERT - Bidirectional Transformers": render_section_4_11,
    }

    sub_selection = st.sidebar.radio(
        "Chapter 4 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch4_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
    
    # Chapter completed
    st.markdown("---")
    st.success("✅ Chapter 4 completed! The Transformer architecture builds on these attention concepts.")
