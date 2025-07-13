# chapter7_0.py
import streamlit as st
import matplotlib.pyplot as plt

def render_7_0():
    """Renders the introduction and roadmap for building a nano-GPT."""
    st.subheader("7.0: The Roadmap - Building a 'nano-GPT'")
    st.markdown("""
    Welcome to the most hands-on chapter yet! We are going to build our very own generative language model from scratch, using the principles of the Transformer architecture.

    **Motivation:** Training a real Large Language Model like GPT-3 costs millions of dollars and requires immense computational power. However, we can build a tiny version, a **"nano-GPT"**, on a small text file to understand every single component that makes these models work. By the end of this chapter, you will have a complete, runnable Python script for your own character-level language model.
    """)

    st.subheader("Our Project Roadmap")
    st.markdown("We will build our model step-by-step:")
    
    # --- Visualization of the Roadmap ---
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    steps = ["1. Data &\nTokenizer", "2. Model\nComponents", "3. Transformer\nBlock", "4. Full\nModel", "5. Training\nLoop", "6. Generate\nText"]
    
    for i, step in enumerate(steps):
        ax.text(i * 1.6 + 1, 0.5, step, ha='center', va='center', size=10, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
        if i < len(steps) - 1:
            ax.arrow(i * 1.6 + 1.6, 0.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# -------------------------------------------------------------------

# chapter7_1.py
import streamlit as st

def render_7_1():
    st.subheader("7.1: Step 1 - The Data and the Tokenizer")
    st.info("Content for this section is coming soon!")

# -------------------------------------------------------------------

# chapter7_2.py
import streamlit as st

def render_7_2():
    st.subheader("7.2: Step 2 - Building the Model Components")
    st.info("Content for this section is coming soon!")

# -------------------------------------------------------------------

# chapter7_3.py
import streamlit as st

def render_7_3():
    st.subheader("7.3: Step 3 - Assembling the Transformer Block")
    st.info("Content for this section is coming soon!")

# -------------------------------------------------------------------

# chapter7_4.py
import streamlit as st

def render_7_4():
    st.subheader("7.4: Step 4 - Creating the Full Model")
    st.info("Content for this section is coming soon!")

# -------------------------------------------------------------------

# chapter7_5.py
import streamlit as st

def render_7_5():
    st.subheader("7.5: Step 5 - The Training Loop")
    st.info("Content for this section is coming soon!")

# -------------------------------------------------------------------

# chapter7_6.py
import streamlit as st

def render_7_6():
    st.subheader("7.6: The Grand Finale - Full Code & Generation")
    st.info("Content for this section is coming soon!")
