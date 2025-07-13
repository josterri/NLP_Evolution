import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_4_5():
    """Renders the interactive attention workbench."""
    st.subheader("4.5: Interactive Attention Workbench")
    st.markdown("""
    Let's put everything we've learned in this chapter together in one interactive session. Here, you can provide your own sentence, select a word, and see the entire self-attention calculation unfold for that word.
    """)

    # --- User Input ---
    sentence = st.text_input("Enter a short sentence:", "the cat sat on the mat")
    tokens = sentence.lower().split()

    if len(tokens) < 2:
        st.warning("Please enter a sentence with at least two words.")
        return

    target_word = st.selectbox("Select a word to analyze its attention mechanism:", options=tokens)
    target_idx = tokens.index(target_word)

    # --- Simulate Q, K, V vectors ---
    st.markdown("---")
    st.subheader("1. Simulated Q, K, V Vectors")
    st.markdown("For this demo, we'll generate random Q, K, and V vectors for each word. In a real model, these would be learned.")

    # Use a seed based on the sentence to keep vectors consistent for the same sentence
    seed = sum(ord(c) for c in sentence)
    np.random.seed(seed)
    embedding_dim = 4 # Use a small dimension for clarity
    
    q_vectors = {word: np.random.rand(embedding_dim) for word in tokens}
    k_vectors = {word: np.random.rand(embedding_dim) for word in tokens}
    v_vectors = {word: np.random.rand(embedding_dim) for word in tokens}
    d_k = embedding_dim

    with st.expander("Show Generated Q, K, V Vectors"):
        st.write("**Query Vectors (Q):**")
        st.json({k: list(np.round(v, 2)) for k, v in q_vectors.items()})
        st.write("**Key Vectors (K):**")
        st.json({k: list(np.round(v, 2)) for k, v in k_vectors.items()})
        st.write("**Value Vectors (V):**")
        st.json({k: list(np.round(v, 2)) for k, v in v_vectors.items()})

    # --- Attention Calculation ---
    st.markdown("---")
    st.subheader(f"2. Attention Calculation for '{target_word}'")

    # Step 1: Score
    query_target = q_vectors[target_word]
    scores = np.array([np.dot(query_target, k_vectors[word]) for word in tokens])
    
    # Step 2: Scale
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    attention_weights = softmax(scaled_scores)
    
    # Step 4: Weighted Sum
    v_matrix = np.array([v_vectors[word] for word in tokens])
    output_vector = np.dot(attention_weights, v_matrix)

    # --- Display Results ---
    st.markdown(f"We use the **Query** from `'{target_word}'` and compare it to the **Key** of every other word.")
    
    results_data = {
        "Word": tokens,
        "Score (Q•K)": scores,
        "Scaled Score": scaled_scores,
        "Attention Weight (Softmax)": attention_weights
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df.style.format({
        "Score (Q•K)": "{:.2f}",
        "Scaled Score": "{:.2f}",
        "Attention Weight (Softmax)": "{:.2%}"
    }).background_gradient(subset=["Attention Weight (Softmax)"], cmap='viridis'))

    st.markdown("#### Final Output Vector")
    st.markdown(f"The final, context-aware vector for `'{target_word}'` is a weighted sum of all Value vectors.")
    st.write(f"**Output Vector for '{target_word}':**")
    st.code(str(np.round(output_vector, 3)))

    st.success("This new vector now contains information about the other words in the sentence, blended according to the attention weights!")
