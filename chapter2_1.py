import streamlit as st
import pandas as pd
import numpy as np

def render_2_1():
    """Renders the One-Hot Encoding section."""
    st.subheader("2.1: Representing Words as Numbers - One-Hot Encoding")
    st.markdown("""
    Before we can give words 'meaning', we first need a way to represent them numerically so a computer can process them. The most basic and traditional method for this is **One-Hot Encoding**.

    The idea is simple: for a given vocabulary of words, we represent each word as a vector of all zeros, except for a single '1' at the index corresponding to that word.
    """)

    st.subheader("🧠 The Theory: A Vector for Every Word")
    st.markdown("""
    Imagine our entire vocabulary is just four words: `["cat", "dog", "mat", "sat"]`.
    - We assign an index to each word: `cat`=0, `dog`=1, `mat`=2, `sat`=3.
    - The one-hot vector for each word is a vector with a length equal to the vocabulary size.
    
    The resulting vectors would be:
    - **cat**: `[1, 0, 0, 0]`
    - **dog**: `[0, 1, 0, 0]`
    - **mat**: `[0, 0, 1, 0]`
    - **sat**: `[0, 0, 0, 1]`
    """)

    st.subheader("Visualizing the Sparsity")
    st.markdown("Here is how the one-hot vectors for our tiny vocabulary would look in a table.")
    
    # --- Visualization Demo ---
    vocab = ["cat", "dog", "mat", "sat"]
    one_hot_vectors = np.identity(len(vocab), dtype=int)
    df = pd.DataFrame(one_hot_vectors, index=vocab, columns=[f"Index {i}" for i in range(len(vocab))])
    st.dataframe(df)

    st.error("""
    **The Problem:** Notice two major flaws:
    1.  **Sparsity & High Dimensionality:** The vectors are mostly zeros. For a real-world vocabulary of 50,000 words, each vector would have 50,000 dimensions, which is computationally very inefficient.
    2.  **No Semantic Relationship:** The vector for "cat" is mathematically as different from "dog" as it is from "mat". The representation contains no information about which words have similar meanings. Every word is an island.
    
    These limitations are precisely what Word Embeddings were designed to solve.
    """)
    
    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Vector Creation:** If you added the word "ran" to our vocabulary, what would its one-hot vector be? What is the new length of all the vectors?
    2.  **Distance:** In this vector space, what is the distance between "cat" and "dog"? What about "cat" and "sat"? What does this tell you about the limitations of this representation?
    """)

    st.subheader("🐍 The Python Behind the Encoding")
    with st.expander("Show the Python Code for One-Hot Encoding"):
        st.code("""
import numpy as np

def one_hot_encode(word, vocabulary):
    # Create a mapping from word to index
    word_to_idx = {w: i for i, w in enumerate(vocabulary)}
    
    # Check if the word is in our vocabulary
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' not in vocabulary.")
        
    # Create a vector of zeros
    vector = np.zeros(len(vocabulary))
    
    # Set the '1' at the correct index
    vector[word_to_idx[word]] = 1
    
    return vector

# --- Example ---
vocab = ["cat", "dog", "mat", "sat"]
cat_vector = one_hot_encode("cat", vocab)
# Result: [1. 0. 0. 0.]
print(cat_vector)
        """, language='python')
