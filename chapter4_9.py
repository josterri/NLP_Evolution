import streamlit as st
import numpy as np
import pandas as pd

def render_4_9():
    """Renders the detailed walkthrough of attention-based prediction."""
    st.subheader("4.9: Attention Prediction - A Detailed Walkthrough")
    st.markdown("""
    Let's dissect the **Attention-based Prediction** from the workbench in the previous section. We will use a fixed example, **"the cat sat on"**, and go through every single calculation to see how the final prediction is made.
    """)

    # --- Shared Setup ---
    st.markdown("---")
    st.subheader("Setup: The 'Knowledge Base'")
    st.markdown("We start with a fixed vocabulary and pre-assigned (random) static embeddings and transformation matrices. In a real scenario, these would be learned.")

    vocab = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'rug']
    embedding_dim = 4
    np.random.seed(42) # for reproducibility
    static_embeddings = {word: np.random.rand(embedding_dim) for word in vocab}
    Wq, Wk, Wv = [np.random.rand(embedding_dim, embedding_dim) for _ in range(3)]
    context_words = ["the", "cat", "sat", "on"]

    with st.expander("Show Initial Setup Values"):
        st.write("**Vocabulary:**", vocab)
        st.write("**Static Embedding for 'cat':**", np.round(static_embeddings['cat'], 2))
        st.write("**Query Matrix (Wq) Shape:**", Wq.shape)

    # --- Step 1: Q, K, V Generation ---
    st.markdown("---")
    st.subheader("Step 1: Generate Q, K, V for the Context")
    st.markdown("For each word in our context `['the', 'cat', 'sat', 'on']`, we transform its static embedding into three specialized vectors.")
    
    qkv_data = {word: (np.dot(static_embeddings[word], Wq), np.dot(static_embeddings[word], Wk), np.dot(static_embeddings[word], Wv)) for word in context_words}
    queries, keys, values = [list(x) for x in zip(*[qkv_data[word] for word in context_words])]
    
    with st.expander("Show Q, K, V Vectors"):
        st.write("The following vectors have been generated from the initial embeddings:")
        q_df = pd.DataFrame(queries, index=context_words, columns=[f"q{i}" for i in range(embedding_dim)])
        k_df = pd.DataFrame(keys, index=context_words, columns=[f"k{i}" for i in range(embedding_dim)])
        v_df = pd.DataFrame(values, index=context_words, columns=[f"v{i}" for i in range(embedding_dim)])
        st.write("Query Vectors (Q):")
        st.dataframe(q_df.style.format("{:.2f}"))
        st.write("Key Vectors (K):")
        st.dataframe(k_df.style.format("{:.2f}"))
        st.write("Value Vectors (V):")
        st.dataframe(v_df.style.format("{:.2f}"))

    # --- Step 2: Score Calculation ---
    st.markdown("---")
    st.subheader("Step 2: Calculate Attention Scores")
    st.markdown("We take the **Query** from the last word (`on`) and calculate its dot product with the **Key** of every word in the context to get a relevance score.")
    
    query_target = queries[-1] # Query for 'on'
    scores = np.array([np.dot(query_target, k) for k in keys])
    
    score_df = pd.DataFrame(scores, index=context_words, columns=["Score (Q • K)"])
    st.dataframe(score_df.style.format("{:.2f}"))

    # --- Step 3: Scale & Softmax ---
    st.markdown("---")
    st.subheader("Step 3: Calculate Final Attention Weights")
    st.markdown("We scale the scores and apply a softmax function to turn them into a probability distribution. These weights determine how much 'focus' to put on each context word.")
    
    scaled_scores = scores / np.sqrt(embedding_dim)
    def softmax(x): return np.exp(x) / np.sum(np.exp(x))
    attention_weights = softmax(scaled_scores)

    weights_df = pd.DataFrame(attention_weights, index=context_words, columns=["Attention Weight"])
    st.write("Attention Weights for `on`:")
    st.dataframe(weights_df.style.format("{:.2%}").background_gradient(cmap='viridis'))

    # --- Step 4: Create Context Vector ---
    st.markdown("---")
    st.subheader("Step 4: Create the Context-Aware Vector")
    st.markdown("We create the final context vector by calculating a weighted sum of all the **Value** vectors, using our attention weights.")
    
    context_vector = np.dot(attention_weights, np.array(values))
    st.write("`Context Vector = (Weight_the * Value_the) + (Weight_cat * Value_cat) + ...`")
    st.write("**Resulting Context Vector:**")
    st.code(np.round(context_vector, 3))
    st.caption("This single vector is a rich summary of the entire context, viewed from the perspective of the word 'on'.")

    # --- Step 5: Final Prediction ---
    st.markdown("---")
    st.subheader("Step 5: Predict the Next Word")
    st.markdown("Finally, we take our new context vector and compare it to the static embeddings of all words in our original vocabulary to find the most similar one.")

    similarities = {}
    for vocab_word, vocab_embedding in static_embeddings.items():
        if vocab_word not in context_words:
            sim = np.dot(context_vector, vocab_embedding) / (np.linalg.norm(context_vector) * np.linalg.norm(vocab_embedding))
            similarities[vocab_word] = sim
            
    prediction = max(similarities, key=similarities.get)
    
    sim_df = pd.DataFrame.from_dict(similarities, orient='index', columns=['Similarity Score']).sort_values(by='Similarity Score', ascending=False)
    st.write("Similarity between Context Vector and Vocabulary:")
    st.dataframe(sim_df.style.format("{:.2f}"))

    st.success(f"The word with the highest similarity to our context vector is **'{prediction}'**. This is our final prediction!")

