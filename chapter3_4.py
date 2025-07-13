import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_3_4():
    """Renders the Attention Mechanism section."""
    st.subheader("3.4: The Attention Mechanism - Focusing on What Matters")
    st.markdown("""
    Sequential models like ELMo were a huge step forward, but their reliance on processing word-by-word was a bottleneck. They also struggled to connect words that were very far apart in a long sentence.

    The next great leap was the **Attention Mechanism**. Instead of a rolling memory, attention allows a model to look at all words in the sentence *simultaneously* and decide which ones are most important for understanding any given word.
    """)

    st.subheader("🧠 The Core Idea: A Weighted Focus")
    st.markdown("""
    Imagine you are translating the sentence: "The black cat sat on the mat". When you get to the word "sat", your brain instinctively pays more attention to "cat" (the subject) than to "black" or "the".

    The Attention mechanism formalizes this. For each word, it calculates an "attention score" to every other word in the sentence. A high score means "pay close attention to this word when trying to understand me." This allows the model to build a rich contextual representation by creating a weighted average of all other words, where the weights are the attention scores.
    """)

    st.subheader("🛠️ Interactive Demo: Visualizing Self-Attention")
    st.markdown("Let's visualize how a word 'pays attention' to other words in a sentence. Enter a sentence and select a word to see its simulated attention scores.")
    
    sentence = st.text_input("Enter a sentence:", "the tired dragon flew over the green meadows")
    tokens = sentence.lower().split()

    if len(tokens) > 1:
        selected_word = st.selectbox("Select a word to analyze its attention:", options=tokens)
        
        if st.button("Calculate Attention Scores"):
            # --- Helper Functions for Attention Demo ---
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)

            def calculate_attention(tokens, selected_word_idx):
                """Simulates a self-attention mechanism."""
                np.random.seed(42) # for reproducibility
                embedding_dim = 8
                # In a real model, these are learned. Here, we simulate them.
                queries = {token: np.random.rand(embedding_dim) for token in set(tokens)}
                keys = {token: vec for token, vec in queries.items()}
                
                query_vector = queries[tokens[selected_word_idx]]
                scores = np.array([np.dot(query_vector, keys[token]) for token in tokens])
                return softmax(scores)

            selected_word_idx = tokens.index(selected_word)
            attention_weights = calculate_attention(tokens, selected_word_idx)
            
            st.write(f"Attention scores for the word **'{selected_word}'**:")

            # Create a DataFrame for visualization
            df = pd.DataFrame([attention_weights], columns=tokens, index=[f"Attention from '{selected_word}'"])

            # Create the heatmap
            fig, ax = plt.subplots()
            sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", ax=ax, cbar=False)
            ax.set_title(f"Attention from '{selected_word}' to other words")
            st.pyplot(fig)

            st.success("""
            The heatmap shows how much the word **'{selected_word}'** focuses on other words (including itself) to build its contextual meaning. A higher score means higher attention. This ability to weigh the importance of all words at once is what makes the Transformer architecture so powerful.
            """)
    else:
        st.warning("Please enter a sentence with at least two words.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Pronoun Resolution:** Try the sentence "The dragon saw the knight and he waved". Select the word "he". In a real model, who would "he" pay the most attention to?
    2.  **Verb-Object Relationship:** Use the sentence "The cat chased the mouse". Select "chased". Which words do you think are most important for defining the action of chasing?
    """)
