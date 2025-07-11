import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_3_1():
    """Renders content for section 3.1."""
    
    # --- Helper Function ---
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
    st.subheader("3.1 The Problem: A Word is Not an Island")
    st.markdown("""
    As we saw, static embeddings (like Word2Vec) were a massive leap forward. They gave models a sense of a word's general meaning. However, they suffer from a fundamental flaw: **they assign only one vector to each word**.

    Language is ambiguous. The same word can have completely different meanings depending on the context. This is the **problem of polysemy**.

    Consider the word "**bank**":
    - "I need to go to the **bank** to deposit money." (a financial institution)
    - "We sat on the river **bank** and watched the boats." (the side of a river)

    A static embedding for "bank" is an awkward average of these two meanings. To truly understand language, a model needs to generate **dynamic, contextual embeddings**—vectors that change based on the surrounding words.
    """)

    st.subheader("🛠️ Interactive Demo: See Context in Action")
    st.markdown("Let's simulate how a model would generate a different vector for the word **bank** depending on the sentence. The plot below will show where the vector for 'bank' lands based on your input.")
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
