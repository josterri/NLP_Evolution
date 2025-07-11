import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_chapter_3():
    """Renders all content for Chapter 3."""

    # --- Helper Functions (Chapter 3) ---
    def get_contextual_embedding_for_bank(sentence):
        sentence = sentence.lower()
        financial_keywords = ['money', 'deposit', 'loan', 'account', 'atm', 'cash']
        river_keywords = ['river', 'water', 'boat', 'fish', 'sand', 'edge']
        is_financial = any(word in sentence for word in financial_keywords)
        is_river = any(word in sentence for word in river_keywords)
        financial_vec = np.array([8, 2])
        river_vec = np.array([2, 8])
        neutral_vec = np.array([5, 5])
        if is_financial and not is_river:
            return financial_vec, "Financial Context Detected"
        elif is_river and not is_financial:
            return river_vec, "Geographical Context Detected"
        else:
            return neutral_vec, "Ambiguous or No Context Detected"

    # --- UI Rendering ---
    st.header("Chapter 3: The Power of Context - Contextual Embeddings")
    st.markdown("""Word embeddings were a massive breakthrough, but they had a crucial flaw: they are **static**. Each word has only one vector, regardless of its context. The next revolution was to create **dynamic, contextual embeddings**—vectors that change based on the surrounding words.""")

    st.subheader("🧠 The Theory: Different Context, Different Vector")
    st.markdown("""Instead of a fixed dictionary lookup, a contextual model reads the *entire sentence* before generating a vector for each word. It pays attention to the other words to understand the intended meaning.""")

    st.subheader("🛠️ Interactive Demo: See Context in Action")
    st.markdown("Let's simulate how a model generates a different vector for the word **bank** depending on the sentence.")
    sentence = st.text_input("Write a sentence containing the word 'bank':", "I need to deposit money at the bank.")

    if st.button("Analyze Sentence Context"):
        if 'bank' not in sentence.lower():
            st.error("Please include the word 'bank' in your sentence.")
        else:
            bank_vector, context_message = get_contextual_embedding_for_bank(sentence)
            st.info(context_message)
            fig, ax = plt.subplots()
            ax.scatter([8], [2], s=100, c='red', marker='X', label='Financial Meaning')
            ax.text(8.1, 2.1, 'Financial "Bank"')
            ax.scatter([2], [8], s=100, c='blue', marker='X', label='River Meaning')
            ax.text(2.1, 8.1, 'River "Bank"')
            ax.scatter([5], [5], s=100, c='gray', marker='o', label='Ambiguous Meaning')
            ax.text(5.1, 5.1, 'Ambiguous "Bank"')
            ax.scatter(bank_vector[0], bank_vector[1], s=200, c='green', marker='*', label='Your Contextual "Bank"')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_title("Contextual Vector for 'Bank'")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.success(f"The generated vector for 'bank' in this context is: `{bank_vector}`")
