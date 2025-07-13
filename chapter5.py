import streamlit as st

# Import the render functions from the sub-chapter files
from chapter5_5 import render_5_5
from chapter5_1 import render_5_1
from chapter5_2 import render_5_2
from chapter5_3 import render_5_3
from chapter5_4 import render_5_4
from chapter5_5 import render_5_5
from chapter5_6 import render_5_6
#from chapter5_1 import render_5_1
#from chapter5_2 import render_5_2
#from chapter5_3 import render_5_3
#from chapter5_4 import render_5_4

def render_chapter_5():
    """Renders all content for Chapter 5 by controlling sub-chapters."""
    st.header("Chapter 5: Applying the Foundations: Text Classification")

    SUB_CHAPTERS = {
        "5.1: What is Text Classification?": render_5_1,
        "5.2: The Statistical Approach (Bag-of-Words)": render_5_2,
        "5.3: The Embedding Approach (Averaging Vectors)": render_5_3,
        "5.4: The Modern Approach (Fine-tuning Transformers)": render_5_4,
        "5.5: Interactive Classification Workbench": render_5_5,
        "5.6: Consolidated Python Code": render_5_6,
    }

    # Use a selectbox for sub-navigation
    sub_selection = st.selectbox(
        "Chapter 5 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch5_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page
    page = SUB_CHAPTERS[sub_selection]
    page()
