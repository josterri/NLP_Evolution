import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_2_5():
    """Renders the Limitations of Static Embeddings section."""
    st.subheader("2.5: Limitations of Static Embeddings - The Polysemy Problem")
    st.markdown("""
    Word embeddings were a massive breakthrough, but they still have a crucial flaw: **they are static**. This means that each word has only *one* vector representation, regardless of how it's used in a sentence.

    This becomes a major issue for words with multiple meanings, a phenomenon known as **polysemy**.
    """)

    st.subheader("🧠 The Theory: One Vector Isn't Enough")
    st.markdown("""
    Consider the word "**bank**":
    - "I need to go to the **bank** to deposit money." (a financial institution)
    - "We sat on the river **bank** and watched the boats." (the side of a river)

    A static embedding model like Word2Vec is forced to learn a single vector for "bank". This vector ends up being a confusing, watered-down average of its different meanings. It's not quite a financial vector, and not quite a geographical one. This ambiguity limits the model's ability to truly understand the nuance of a sentence.
    """)

    st.subheader("Visualizing the Ambiguity")
    st.markdown("Let's visualize where different concepts might live in a vector space, and the problem this creates for a word like 'bank'.")

    # --- Visualization Demo ---
    financial_words = {'money': np.array([8, 2]), 'cash': np.array([8.5, 2.5]), 'loan': np.array([7.5, 1.5])}
    river_words = {'river': np.array([2, 8]), 'water': np.array([1.5, 8.5]), 'boat': np.array([2.5, 7.5])}
    
    # The problematic, averaged vector for 'bank'
    static_bank_vec = np.mean(list(financial_words.values()) + list(river_words.values()), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot financial cluster
    for word, vec in financial_words.items():
        ax.scatter(vec[0], vec[1], s=100, color='red')
        ax.text(vec[0] + 0.1, vec[1] + 0.1, word, fontsize=12)
    ax.text(8, 1, "Financial Cluster", color='red')

    # Plot river cluster
    for word, vec in river_words.items():
        ax.scatter(vec[0], vec[1], s=100, color='blue')
        ax.text(vec[0] + 0.1, vec[1] + 0.1, word, fontsize=12)
    ax.text(2, 9, "River Cluster", color='blue')

    # Plot the ambiguous 'bank' vector
    ax.scatter(static_bank_vec[0], static_bank_vec[1], s=200, marker='X', color='purple', label='Static vector for "bank"')
    ax.text(static_bank_vec[0] + 0.1, static_bank_vec[1] + 0.1, '"bank"?', fontsize=14, color='purple')

    ax.set_title("The Problem of a Single Vector for 'Bank'")
    ax.set_xlabel("Dimension 1 (e.g. Finance)")
    ax.set_ylabel("Dimension 2 (e.g. Geography)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.error("""
    The static vector for "bank" lies awkwardly between the two distinct meaning clusters. It doesn't accurately represent either context, which confuses the model.
    """)

    st.subheader("The Path Forward")
    st.markdown("""
    To solve this problem, we need a model that can generate **different embeddings** for the same word depending on the sentence it appears in. This is the challenge of **contextualization**, and it's the topic we will explore in the next chapter.
    """)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Find Other Words:** What are some other common English words that would suffer from the polysemy problem? (e.g., "right", "lead", "bat").
    2.  **Sentence Pairs:** Write two sentences where the word "light" has completely different meanings. How would a static embedding fail to capture this difference?
    """)
