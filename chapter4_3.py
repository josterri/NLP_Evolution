import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def render_4_3():
    """Renders the detailed Query, Key, and Value section."""
    st.subheader("4.3: Query, Key, and Value in Detail")
    st.markdown("""
    In the last section, we introduced the library analogy for Attention. Now, let's look at how this is implemented. The model doesn't use the word's original embedding directly for the attention calculation. Instead, it creates three new, specialized vectors from it.

    For every single word in the input sentence, its initial embedding is passed through three separate, small neural networks (linear layers). The outputs of these layers are the **Query**, **Key**, and **Value** vectors.
    """)

    # --- Visualization of QKV creation ---
    st.subheader("Visualizing the Q, K, V Transformation")
    st.markdown("A single input embedding is transformed into three specialized vectors. These transformation 'lenses' (the weight matrices Wq, Wk, Wv) are learned during the model's training.")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Input Embedding
    ax.text(1, 2, "Input Embedding\nfor 'cats'", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightgray"))

    # Transformation matrices (lenses)
    ax.text(4, 3, "Wq Matrix\n(Query Lens)", ha='center', va='center', bbox=dict(boxstyle="sawtooth,pad=0.5", fc="lightgreen"))
    ax.text(4, 2, "Wk Matrix\n(Key Lens)", ha='center', va='center', bbox=dict(boxstyle="sawtooth,pad=0.5", fc="lightyellow"))
    ax.text(4, 1, "Wv Matrix\n(Value Lens)", ha='center', va='center', bbox=dict(boxstyle="sawtooth,pad=0.5", fc="lightpink"))

    # Output vectors
    ax.text(7, 3, "Query Vector (Q)\n'I am a noun, I need a verb'", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
    ax.text(7, 2, "Key Vector (K)\n'I am a plural noun'", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow"))
    ax.text(7, 1, "Value Vector (V)\n(Rich meaning of 'cats')", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightpink"))

    # Arrows
    ax.arrow(2, 2, 1, 0.8, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(2, 2, 1.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(2, 2, 1, -0.8, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5, 3, 1, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5, 2, 1, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5, 1, 1, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    st.pyplot(fig)

    st.subheader("🛠️ Interactive Demo: The Query-Key Matchup")
    st.markdown("Let's simulate how a Query from one word finds the best Key from a set of other words. The dot product between the Query and Key vectors gives us a 'compatibility score'.")

    # --- Interactive Demo ---
    database = {
        "cats":  {"key": np.array([0.1, 0.9]), "value": "a furry, four-legged animal"},
        "were":  {"key": np.array([0.9, 0.1]), "value": "a plural verb"},
        "tired": {"key": np.array([0.5, 0.5]), "value": "a state of being"},
    }
    
    # Simulate a query for the word 'were'
    query_word = "were"
    query_vec = np.array([0.2, 0.8]) # A query from a verb looking for a noun subject

    st.info(f"Let's find the context for the word **'{query_word}'**. Its Query vector is simulated as `{np.round(query_vec, 2)}` (meaning it's looking for a noun).")

    scores = {}
    for word, data in database.items():
        key_vec = data["key"]
        score = np.dot(query_vec, key_vec)
        scores[word] = score

    results_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Compatibility Score'])
    results_df['Key Vector'] = [str(np.round(database[word]['key'], 2)) for word in results_df.index]
    
    st.write("Comparing the Query from 'were' to the Keys of all words:")
    st.dataframe(results_df)
    
    best_match = max(scores, key=scores.get)
    st.success(f"The highest compatibility score is with **'{best_match}'**. The model has learned that the verb 'were' should pay most attention to the noun 'cats' to understand its context.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **The Value Vector:** In our demo, we only used the Query and Key to find the best match. What is the purpose of the third vector, the Value? Why don't we just use the information from the Key?
    2.  **Symmetry:** Is the score from `Query('cats') • Key('were')` the same as `Query('were') • Key('cats')`? Why is this important?
    """)

    st.subheader("🐍 The Python Behind the Transformation")
    with st.expander("Show the Python Code for Q, K, V Generation"):
        st.code("""
import numpy as np

# In a real model, these are learned during training
embedding_size = 128
W_query = np.random.rand(embedding_size, embedding_size)
W_key = np.random.rand(embedding_size, embedding_size)
W_value = np.random.rand(embedding_size, embedding_size)

def get_qkv(input_embedding):
    # The core transformation: a simple matrix multiplication
    query = np.dot(input_embedding, W_query)
    key = np.dot(input_embedding, W_key)
    value = np.dot(input_embedding, W_value)
    return query, key, value

# --- Example ---
# Get the initial embedding for the word 'cats'
cats_embedding = np.random.rand(embedding_size) 

# Transform it into three specialized vectors
q_cats, k_cats, v_cats = get_qkv(cats_embedding)

print(f"Shape of Query vector: {q_cats.shape}")
print(f"Shape of Key vector:   {k_cats.shape}")
print(f"Shape of Value vector: {v_cats.shape}")
        """, language='python')
