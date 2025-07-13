import streamlit as st

# Import the render functions from the sub-chapter files
from chapter2_1 import render_2_1
from chapter2_2 import render_2_2
from chapter2_3 import render_2_3
from chapter2_4 import render_2_4
from chapter2_5 import render_2_5
from chapter2_6 import render_2_6
from chapter2_7 import render_2_7
from chapter2_8 import render_2_8
from chapter2_9 import render_2_9
from chapter2_10 import render_2_10

def render_chapter_2():
    """Renders all content for Chapter 2 by controlling sub-chapters."""
    st.header("Chapter 2: The Rise of Neural Networks & Embeddings")

    SUB_CHAPTERS = {
        "2.1: One-Hot Encoding": render_2_1,
        "2.2: The Concept of Word Embeddings": render_2_2,
        "2.3: Exploring the Vector Space & Analogies": render_2_3,
        "2.4: Limitations of Static Embeddings": render_2_4,
        "2.5: GloVe - An Alternative Appproach": render_2_5,
        "2.6: Demo - Word2Vec in Action": render_2_6,
        "2.7: Under the Hood - Training Word2Vec (Experts)": render_2_7,
        "2.8: Demo - Using Embeddings to Predict a Sequence": render_2_8,
        "2.9: The Power of Pre-trained Models": render_2_9,
        "2.10: A Simple Analogy for Word2Vec": render_2_10,
        
        
        
        
        
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
