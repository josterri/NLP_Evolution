# chapter5_2.py
import streamlit as st
from collections import Counter
import re
import pandas as pd

def render_5_2():
    """Renders the Bag-of-Words section."""
    st.subheader("5.2: The Statistical Approach (Bag-of-Words)")
    st.markdown("""
    The earliest approach to text classification was purely statistical. It completely ignores grammar and word order, treating a sentence as just a **"bag" of its words**.

    The core idea is that the *frequency* of certain words can reliably predict the category. For example, an email with many occurrences of "free", "win", and "prize" is more likely to be spam.
    """)

    st.subheader("🧠 The Method: Counting Words")
    st.markdown("""
    1.  **Create a Vocabulary:** First, we build a vocabulary of all unique words from our entire training dataset.
    2.  **Create a Vector:** For each sentence, we create a vector that is the same length as the vocabulary. Each element in the vector corresponds to a word in the vocabulary.
    3.  **Count Frequencies:** We fill the vector by counting how many times each word from the vocabulary appears in our sentence.

    This vector, which represents the word counts, is then fed into a simple statistical model (like Naive Bayes) that learns to associate certain word counts with certain labels.
    """)

    st.subheader("🛠️ Interactive Demo: Create a Bag-of-Words Vector")
    st.markdown("Enter a sentence below to see it converted into a word count vector based on a fixed vocabulary.")
    
    vocab = sorted(['the', 'cat', 'dog', 'sat', 'on', 'mat', 'was', 'happy'])
    sentence = st.text_input("Enter a sentence:", "The cat sat on the mat, the cat was happy.")
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    
    if tokens:
        word_counts = Counter(tokens)
        vector = [word_counts.get(word, 0) for word in vocab]
        
        df = pd.DataFrame([vector], columns=vocab, index=["Word Count Vector"])
        st.write("Vocabulary:", vocab)
        st.dataframe(df)
    
    st.error("**Limitation:** This method has no concept of meaning. It doesn't know that 'good' and 'great' are similar, or that 'not good' is the opposite of 'good'.")

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Code for Bag-of-Words"):
        st.code("""
from collections import Counter
import re

def create_bow_vector(sentence, vocabulary):
    # Tokenize and count words in the input sentence
    tokens = re.findall(r'\\b\\w+\\b', sentence.lower())
    word_counts = Counter(tokens)
    
    # Create the vector by looking up counts for each word in the vocabulary
    vector = [word_counts.get(word, 0) for word in vocabulary]
    
    return vector

# --- Example ---
vocab = ['the', 'quick', 'brown', 'fox', 'jumps']
sentence = "the fox is quick, the fox is brown"
bow_vector = create_bow_vector(sentence, vocab)
# Result: [2, 1, 1, 2, 0]
print(bow_vector)
        """, language='python')
