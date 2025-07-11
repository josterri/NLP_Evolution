import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_3_3():
    """Renders content for section 3.3."""

    # --- Helper Functions for Attention Demo ---
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def calculate_attention(tokens, selected_word_idx):
        """
        Simulates a self-attention mechanism.
        In a real Transformer, these vectors would be learned. Here, we create them.
        """
        # Create mock Query, Key, Value vectors for each token
        np.random.seed(42) # for reproducibility
        embedding_dim = 8
        embeddings = {token: np.random.rand(embedding_dim) for token in set(tokens)}

        # Simplified: Q, K, V are derived directly from embeddings for this demo
        queries = {token: vec for token, vec in embeddings.items()}
        keys = {token: vec for token, vec in embeddings.items()}
        
        # Get the query vector for our selected word
        query_vector = queries[tokens[selected_word_idx]]

        # Calculate dot-product scores between the selected word's query and all other words' keys
        scores = np.array([np.dot(query_vector, keys[token]) for token in tokens])
        
        # Normalize scores to get attention weights (probabilities)
        attention_weights = softmax(scores)
        
        return attention_weights

    # --- UI Rendering ---
    st.subheader("3.3 The Parallel Revolution: Attention & Transformers")
    st.markdown("""
    ELMo was powerful, but its sequential nature (reading word-by-word with RNNs) was a bottleneck. The next, and most important, revolution was the **Transformer architecture**, introduced in the paper "Attention Is All You Need".

    Transformers abandoned sequential processing entirely, opting for a parallel approach that could analyze all words in a sentence at once. The mechanism that made this possible is **Self-Attention**.
    """)

    st.subheader("🧠 The Theory: 'Attention Is All You Need'")
    st.markdown("""
    Instead of just looking at its immediate neighbors, the self-attention mechanism allows a word to look at **all other words** in the input sentence simultaneously and decide which ones are most important for understanding its own meaning in this specific context.

    For each word, the model generates three vectors:
    - **Query (Q):** Represents the current word's question, "Who am I in this context?"
    - **Key (K):** Represents the word's label or identity, answering "Here is what I am."
    - **Value (V):** Represents the actual meaning or content of the word.

    To find the attention score for a target word, its **Query** vector is compared (via dot product) with the **Key** vector of every other word in the sentence. These scores are then normalized (using a softmax function) to create a set of attention weights. These weights determine how much of each word's **Value** vector should be blended into the target word's final, context-aware representation.
    """)

    st.subheader("🛠️ Interactive Demo: Visualizing Self-Attention")
    st.markdown("Let's visualize how a word 'pays attention' to other words in a sentence. Enter a sentence and select a word to see its attention scores.")
    
    sentence = st.text_input("Enter a sentence:", "The tired dragon flew over the green meadows")
    tokens = sentence.split()

    if len(tokens) > 1:
        selected_word = st.selectbox("Select a word to analyze its attention:", options=tokens)
        selected_word_idx = tokens.index(selected_word)

        if st.button("Calculate Attention Scores"):
            attention_weights = calculate_attention(tokens, selected_word_idx)
            
            st.write(f"Attention scores for the word **'{selected_word}'**:")

            # Create a DataFrame for visualization
            df = pd.DataFrame([attention_weights], columns=tokens, index=[f"Attention from '{selected_word}'"])

            # Create the heatmap
            fig, ax = plt.subplots()
            sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", ax=ax, cbar=False)
            ax.set_title(f"Attention from '{selected_word}' to other words")
            st.pyplot(fig)

            st.markdown(f"""
            The heatmap shows how much the word **'{selected_word}'** focuses on other words (including itself) to build its contextual meaning. A higher score means higher attention.
            In a real Transformer, these scores would be much more nuanced, often focusing on related verbs, subjects, or objects.
            """)
    else:
        st.warning("Please enter a sentence with at least two words.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Pronoun Resolution:** Try the sentence "The dragon saw the knight and he waved". Select the word "he". In a real model, who would "he" pay the most attention to?
    2.  **Verb-Object Relationship:** Use the sentence "The cat chased the mouse". Select "chased". Which words do you think are most important for defining the action of chasing?
    3.  **Adjectives:** Try "The big red ball bounced high". Select "ball". Which words help define what kind of ball it is?
    """)
