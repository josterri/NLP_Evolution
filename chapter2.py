import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_chapter_2():
    """Renders all content for Chapter 2."""

    # --- Helper Functions (Chapter 2) ---
    def get_embedding_vocab():
        return {
            'king': np.array([9.5, 9]), 'queen': np.array([9.5, 1]),
            'man': np.array([8.5, 9]), 'woman': np.array([8.5, 1]),
            'prince': np.array([7.5, 8.5]), 'princess': np.array([7.5, 1.5]),
            'apple': np.array([1, 5]), 'orange': np.array([1, 4]),
            'banana': np.array([1.5, 3.5]), 'grape': np.array([0.5, 3]),
            'dog': np.array([6, 8]), 'cat': np.array([6, 7]),
            'puppy': np.array([5, 8.5]), 'kitten': np.array([5, 7.5]),
            'strong': np.array([9, 6]), 'fast': np.array([8, 6]),
            'sweet': np.array([2, 2]), 'sour': np.array([2, 1])
        }

    # --- UI Rendering ---
    st.header("Chapter 2: Giving Words Meaning - Word Embeddings")
    st.markdown("The next leap was to represent words based on their **meaning and context** using **Word Embeddings**.")
    st.subheader("🧠 The Theory: Words as Vectors")
    st.markdown("""The core idea is to represent each word as a list of numbers (a **vector**), calculated so that words with similar meanings have similar vectors. This allows for **vector arithmetic**: `vector('king') - vector('man') + vector('woman') ≈ vector('queen')`.""")

    st.subheader("🛠️ Interactive Demo: Exploring a Word Vector Space")
    vocab = get_embedding_vocab()
    words = list(vocab.keys())
    fig, ax = plt.subplots(figsize=(10, 8))
    vectors = np.array(list(vocab.values()))
    ax.scatter(vectors[:, 0], vectors[:, 1], s=50)
    for word, (x, y) in vocab.items():
        ax.text(x + 0.1, y + 0.1, word, fontsize=12)
    ax.set_title("A 2D Map of Word Meanings")
    st.pyplot(fig)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Vector Arithmetic:** On the map, the vector for 'king' is `[9.5, 9]` and 'man' is `[8.5, 9]`. The difference is `[1, 0]`. What does this `[1, 0]` vector represent? (Hint: Look at 'queen' vs 'woman').
    2.  **Placement:** Where would you place the word 'delicious' on this map? Near 'sweet' and 'sour', or somewhere else? What about 'car' and 'truck'?
    """)

    st.subheader("🐍 The Python Behind the Demo")
    with st.expander("Show the Python Code for Finding Similar Words"):
        st.code("""
import numpy as np

def find_closest_words(selected_word, vocab, n=3):
    selected_vector = vocab[selected_word]
    distances = []
    for word, vector in vocab.items():
        if word == selected_word: continue
        distance = np.linalg.norm(selected_vector - vector)
        distances.append((word, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:n]
        """, language='python')
