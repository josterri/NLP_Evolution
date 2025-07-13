import streamlit as st
import matplotlib.pyplot as plt

def render_4_1():
    """Renders the motivation for Attention section."""
    st.subheader("4.1: The Motivation for Attention")
    
    st.subheader("Where We've Been: A Quick Recap")
    st.markdown("""
    Let's quickly summarize our journey so far:
    1.  **Chapter 1 (N-grams):** We learned to predict the next word by counting its neighbors. This was simple, but it couldn't handle new phrases and had no sense of word meaning.
    2.  **Chapter 2 (Embeddings):** We solved the meaning problem by representing words as vectors. We learned that words with similar meanings have similar vectors, allowing for amazing analogies. However, these vectors were static and couldn't handle words with multiple meanings.
    3.  **Chapter 3 (Sequential Models):** We solved the multiple-meaning problem by creating contextual embeddings. By reading a sentence word-by-word, models like ELMo could generate a unique vector for a word based on its context.
    """)

    st.subheader("The New Problem: The Information Bottleneck")
    st.markdown("""
    Sequential models, while powerful, created a new problem: an **information bottleneck**. They must compress the meaning of an entire sentence, no matter how long, into a single, fixed-size memory vector.

    For a long sentence like:
    > "The cats, which were sitting by the window in the warm afternoon sun, were tired."

    When the model gets to the word "were", the most important piece of context is the word "cats". But "cats" is many steps away in the sequence. The model's memory of "cats" might have been diluted by all the words in between ("window", "sun", etc.). This makes it difficult to learn long-range dependencies.
    """)

    # --- Visualization of the Bottleneck ---
    st.subheader("Visualizing the Bottleneck")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('off')
    words = ["The", "cats", "...", "sun,", "were", "tired."]
    positions = [1, 2, 3.5, 5, 6, 7]
    for i, word in enumerate(words):
        ax.text(positions[i], 1.5, word, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="lightblue"))
        if i > 0:
            ax.arrow(positions[i-1] + 0.5, 1.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    ax.text(4, 0.5, "Single Memory Vector\n(Information from 'cats' is diluted)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral"))
    ax.arrow(2, 1.3, 2, -0.6, head_width=0.1, head_length=0.1, fc='gray', ec='gray' )
    ax.arrow(5, 1.3, -1, -0.6, head_width=0.1, head_length=0.1, fc='gray', ec='gray')
    st.pyplot(fig)

    st.error("**The core motivation for Attention was to break this bottleneck.** Instead of forcing all information through a single, sequential memory state, what if we could create direct shortcuts between words, no matter how far apart they are?")
    st.success("Attention allows the word 'were' to directly 'look at' and connect with the word 'cats', solving the long-range dependency problem.")

