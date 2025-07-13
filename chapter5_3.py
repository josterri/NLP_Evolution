# chapter5_3.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_5_3():
    """Renders the Averaging Vectors section."""
    st.subheader("5.3: The Embedding Approach (Averaging Vectors)")
    st.markdown("""
    The Bag-of-Words approach fails to capture the meaning of words. The next logical step was to use the **word embeddings** we learned about in Chapter 2.

    Instead of a sparse vector of word counts, we can create a single, dense vector that represents the overall meaning of the sentence.
    """)

    st.subheader("🧠 The Method: Averaging Word Vectors")
    st.markdown("""
    The simplest way to do this is to:
    1.  Get the pre-trained word embedding (e.g., from Word2Vec or GloVe) for every word in the sentence.
    2.  Average these vectors together element-wise.

    The resulting averaged vector is a single, dense representation of the sentence's meaning. This 'sentence embedding' can then be fed into a classifier. This is a huge improvement because it understands that sentences like "The movie was great" and "The film was fantastic" should have very similar vectors.
    """)

    st.subheader("Visualizing the Averaging Process")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    # Word vectors
    ax.text(0.5, 0.8, "The", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
    ax.text(2.0, 0.8, "movie", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
    ax.text(3.5, 0.8, "was", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
    ax.text(5.0, 0.8, "great", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))

    # Arrows pointing down
    ax.arrow(0.5, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(2.0, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(3.5, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5.0, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Averaging box
    ax.text(2.75, 0.2, "Average all vectors", ha='center', va='center', size=12, bbox=dict(boxstyle="sawtooth,pad=0.5", fc="lightgray"))
    ax.arrow(2.75, 0.1, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Final sentence vector
    ax.text(2.75, -0.2, "Sentence Embedding", ha='center', va='center', size=14, color="green", bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
    st.pyplot(fig)
    
    st.error("**Limitation:** Averaging all the word vectors together loses all information about word order and grammar. The sentences 'The cat chased the dog' and 'The dog chased the cat' would have the exact same sentence embedding.")

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Code for Averaging Embeddings"):
        st.code("""
import numpy as np

def create_sentence_embedding(sentence, embedding_model):
    tokens = sentence.lower().split()
    
    # Get the vector for each word in the sentence, if it exists in the model
    word_vectors = [embedding_model[word] for word in tokens if word in embedding_model]
    
    if not word_vectors:
        # If no words are in the model, return a zero vector
        # You need to know the embedding dimension beforehand
        embedding_dim = 100 # Example dimension
        return np.zeros(embedding_dim)
        
    # Calculate the mean of all word vectors
    sentence_vector = np.mean(word_vectors, axis=0)
    
    return sentence_vector

# --- Example ---
# Assume `model` is a pre-trained Word2Vec or GloVe model
# sentence = "the movie was great"
# sentence_embedding = create_sentence_embedding(sentence, model)
# print(sentence_embedding.shape) # e.g., (100,)
        """, language='python')

# -------------------------------------------------------------------
