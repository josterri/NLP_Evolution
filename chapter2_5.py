import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

def render_2_5():
    """Renders the GloVe section."""
    st.subheader("2.5: GloVe - An Alternative Approach")
    st.markdown("""
    While Word2Vec was taking the world by storm with its predictive approach, researchers at Stanford developed a different, powerful method for creating static embeddings: **GloVe (Global Vectors for Word Representation)**.

    Instead of predicting context words, GloVe is a **count-based model**. It operates on the principle that you can derive meaning directly from the global co-occurrence statistics of words across an entire corpus.
    """)

    st.subheader("🧠 The Theory: Ratios of Probabilities")
    st.markdown("""
    GloVe's core insight is that the *ratio* of co-occurrence probabilities can reveal interesting relationships. For example:
    - The ratio of `P(k | ice) / P(k | steam)` will be large, because "ice" co-occurs with words `k` like "solid" far more than "steam" does.
    - The ratio for a word like "water" (related to both) or "fashion" (related to neither) will be close to 1.

    GloVe builds a large **co-occurrence matrix** that captures how often each word appears in the context of every other word. It then uses matrix factorization to learn lower-dimensional vectors (the embeddings) that best explain these co-occurrence ratios.
    """)

    st.subheader("Visualizing a Co-occurrence Matrix")
    st.markdown("Let's build a simple co-occurrence matrix from a small text with a window size of 1.")
    
    # --- Visualization Demo ---
    text = "I love my cat I love my dog"
    tokens = text.lower().split()
    vocab = sorted(list(set(tokens)))
    
    co_occurrence = pd.DataFrame(0, index=vocab, columns=vocab)

    for i, word in enumerate(tokens):
        if i > 0: # Left context
            context_word = tokens[i-1]
            co_occurrence.loc[word, context_word] += 1
        if i < len(tokens) - 1: # Right context
            context_word = tokens[i+1]
            co_occurrence.loc[word, context_word] += 1
            
    st.write("Our training text is:")
    st.info(f"`{text}`")
    st.write("Resulting Co-occurrence Matrix:")
    st.dataframe(co_occurrence)
    st.success("GloVe learns its vectors by factorizing a much larger version of this matrix, aiming to preserve the ratios of these counts.")


    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Word2Vec vs. GloVe:** What is the fundamental difference in the training objective between Word2Vec (predictive) and GloVe (count-based)?
    2.  **"Global":** Why is the word "Global" in the name GloVe? (Hint: How does it gather its statistics compared to Word2Vec's local context windows?)
    3.  **Window Size:** How would the co-occurrence matrix above change if we used a window size of 2 instead of 1?
    """)

    st.subheader("🐍 The Python Behind the Matrix")
    with st.expander("Show the Python Code for Building a Co-occurrence Matrix"):
        st.code("""
import pandas as pd
import numpy as np

def build_co_occurrence_matrix(tokens, window_size=1):
    vocabulary = sorted(list(set(tokens)))
    co_occurrence_matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary)

    for i, target_word in enumerate(tokens):
        # Define the context window
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if i == j:
                continue
            context_word = tokens[j]
            co_occurrence_matrix.loc[target_word, context_word] += 1
            
    return co_occurrence_matrix

# --- Example ---
text = "I love my cat"
tokens = text.lower().split()
matrix = build_co_occurrence_matrix(tokens, window_size=1)
print(matrix)
        """, language='python')
