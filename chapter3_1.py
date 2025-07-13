import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_3_1():
    """Renders the Polysemy Problem section."""
    st.subheader("3.1: The Limits of a Dictionary (The Polysemy Problem)")
    st.markdown("""
    In the last chapter, we saw the power of word embeddings. They act like a sophisticated dictionary, where each word has a single, fixed entry (its vector). But what happens when a word has multiple definitions?

    This is the **problem of polysemy**, and it's the critical limitation of static embeddings like Word2Vec and GloVe.
    """)

    st.subheader("🧠 The Core Problem: One Word, Many Meanings")
    st.markdown("""
    Consider the word "**bat**":
    - "He swung the **bat** and hit a home run." (a piece of sports equipment)
    - "A **bat** flew out of the cave at dusk." (a nocturnal flying mammal)

    A static embedding model is forced to learn a single vector for "bat". This vector ends up being a confusing average of its different meanings. It's not quite a sports vector, and not quite an animal vector. This ambiguity limits the model's ability to truly understand the nuance of a sentence.
    """)

    st.subheader("Visualizing the Ambiguity")
    st.markdown("Let's visualize where different concepts might live in a vector space, and the problem this creates for a word like 'bat'.")

    # --- Visualization Demo ---
    sports_words = {'ball': np.array([8, 2]), 'hit': np.array([8.5, 2.5]), 'game': np.array([7.5, 1.5])}
    animal_words = {'fly': np.array([2, 8]), 'animal': np.array([1.5, 8.5]), 'cave': np.array([2.5, 7.5])}
    
    # The problematic, averaged vector for 'bat'
    static_bat_vec = np.mean(list(sports_words.values()) + list(animal_words.values()), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot sports cluster
    for word, vec in sports_words.items():
        ax.scatter(vec[0], vec[1], s=100, color='red')
        ax.text(vec[0] + 0.1, vec[1] + 0.1, word, fontsize=12)
    ax.text(8, 1, "Sports Cluster", color='red')

    # Plot animal cluster
    for word, vec in animal_words.items():
        ax.scatter(vec[0], vec[1], s=100, color='blue')
        ax.text(vec[0] + 0.1, vec[1] + 0.1, word, fontsize=12)
    ax.text(2, 9, "Animal Cluster", color='blue')

    # Plot the ambiguous 'bat' vector
    ax.scatter(static_bat_vec[0], static_bat_vec[1], s=200, marker='X', color='purple', label='Static vector for "bat"')
    ax.text(static_bat_vec[0] + 0.1, static_bat_vec[1] + 0.1, '"bat"?', fontsize=14, color='purple')

    ax.set_title("The Problem of a Single Vector for 'Bat'")
    ax.set_xlabel("Dimension 1 (e.g. Sports)")
    ax.set_ylabel("Dimension 2 (e.g. Animals)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.error("""
    The static vector for "bat" lies awkwardly between the two distinct meaning clusters. It doesn't accurately represent either context, which confuses the model. To truly understand language, we need embeddings that can change.
    """)
