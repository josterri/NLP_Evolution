import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def render_2_2():
    """Renders the Word Embeddings section."""
    st.subheader("2.2: Word Embeddings - Giving Words Meaning")
    st.markdown("""
    The limitations of N-grams (sparsity, lack of generalization) led to a major paradigm shift in NLP. Instead of treating words as discrete, isolated symbols, researchers began to represent them as dense vectors in a multi-dimensional space. This is the concept of **Word Embeddings**.

    The core idea is based on the **Distributional Hypothesis**: "a word is characterized by the company it keeps." In other words, words that appear in similar contexts tend to have similar meanings. Word embeddings aim to capture these relationships numerically.
    """)

    st.subheader("🧠 The Theory: From One-Hot to Dense Vectors")
    st.markdown("""
    Previously, a word could be represented by a **one-hot vector**—a huge vector of all zeros except for a single '1' at the index corresponding to that word. This approach is sparse, inefficient, and treats every word as being equally different from every other word.

    Word embeddings, like those produced by the **Word2Vec** model, are **dense vectors**. They are typically much shorter (e.g., 300 dimensions instead of 50,000) and every element is a floating-point number. These vectors are learned, not manually assigned, and their values place words with similar meanings close to each other in the vector space.
    """)
    
    st.subheader("Visualizing a Vector Space")
    st.markdown("Imagine a 2D 'map' of word meanings. Words are not just random points; their location is meaningful. Let's explore a simplified version.")

    # --- Visualization Demo ---
    vocab = {
        'king': np.array([9.5, 9]), 'queen': np.array([9.5, 1]),
        'man': np.array([8.5, 9]), 'woman': np.array([8.5, 1]),
        'apple': np.array([1, 5]), 'orange': np.array([1, 4]),
        'strong': np.array([9, 6]), 'fast': np.array([8, 6]),
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for word, vec in vocab.items():
        ax.scatter(vec[0], vec[1], s=100)
        ax.text(vec[0] + 0.1, vec[1] + 0.1, word, fontsize=12)

    ax.set_title("A 2D Map of Word Meanings")
    ax.set_xlabel("Dimension 1 (e.g., 'Power/Royalty')")
    ax.set_ylabel("Dimension 2 (e.g., 'Gender/Concept')")
    ax.grid(True)
    st.pyplot(fig)

    st.success("""
    Notice the relationships. The vector from 'man' to 'king' is similar to the vector from 'woman' to 'queen'. This allows for amazing **vector arithmetic**, like the famous example:
    `vector('king') - vector('man') + vector('woman') ≈ vector('queen')`
    """)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Find the Relationship:** On the map, what might the vector difference between 'king' and 'queen' represent?
    2.  **Placement:** Where would you place the word 'prince' on this map? What about 'banana'?
    3.  **New Dimensions:** Our map has 2 dimensions. Real word embeddings have hundreds. What kind of relationships could a 3rd, 4th, or 100th dimension capture? (e.g., verb tense, plurality, abstractness).
    """)
