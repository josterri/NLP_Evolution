import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def render_4_8():
    """Renders the interactive code demonstration section."""
    st.subheader("4.8: Interactive Code Workbench")
    st.markdown("""
    This section provides an interactive workbench to explore the three prediction methods from the cumulative exercise. You can provide a small corpus and a seed phrase, and then see how each method arrives at its prediction for the **next single word**.
    """)

    # --- Shared Setup ---
    st.markdown("---")
    st.subheader("Setup: The Knowledge Base")
    text = st.text_area("Provide a small text corpus:",
                        "the cat sat on the mat . the dog sat on the rug .",
                        height=100)
    tokens = text.lower().replace('.', '').replace(',', '').split()
    vocab = sorted(list(set(tokens)))

    # --- Interactive Demo 1: N-gram Prediction ---
    st.markdown("---")
    st.subheader("1. N-gram Prediction Workbench")
    st.markdown("This method uses frequency counts of word pairs (bigrams) or triplets (trigrams) to predict the most likely next word.")
    
    ngram_sequence = st.text_input("Enter a seed phrase for N-gram:", "the cat")
    if st.button("Predict with N-gram"):
        # Build a simple trigram model
        model = defaultdict(Counter)
        for i in range(len(tokens) - 3 + 1):
            context, target = tuple(tokens[i:i+2]), tokens[i+2]
            model[context][target] += 1
        
        context = tuple(ngram_sequence.lower().split())
        st.write(f"**1. Context:** The model looks at the last two words: `{context}`.")
        
        if context in model:
            prediction = model[context].most_common(1)[0][0]
            st.write(f"**2. Lookup:** It finds all words that followed this context in the text: `{dict(model[context])}`.")
            st.success(f"**3. Prediction:** The most frequent word is **'{prediction}'**.")
        else:
            st.error("This exact context was not found in the text.")

    with st.expander("Show N-gram Prediction Code"):
        st.code("""
from collections import Counter, defaultdict

def predict_with_ngram(tokens, sequence, n=3):
    # Build a simple model on the fly
    model = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i+n-1])
        target = tokens[i+n-1]
        model[context][target] += 1
        
    context = tuple(sequence[-(n-1):])
    if context in model:
        return model[context].most_common(1)[0][0]
    return None
        """, language='python')

    # --- Interactive Demo 2: Static Embedding Prediction ---
    st.markdown("---")
    st.subheader("2. Static Embedding Prediction Workbench")
    st.markdown("This method only looks at the last word, finds its vector, and then searches for the most similar vector in the entire vocabulary.")

    static_sequence = st.text_input("Enter a seed phrase for Static Embedding:", "the dog")
    if st.button("Predict with Static Embedding"):
        # Simulate embeddings
        np.random.seed(42)
        static_embeddings = {word: np.random.rand(8) for word in vocab}
        
        last_word = static_sequence.lower().split()[-1]
        st.write(f"**1. Context:** The model only sees the last word: `{last_word}`.")

        if last_word in static_embeddings:
            last_word_embedding = static_embeddings[last_word]
            
            similarities = {}
            for vocab_word, vocab_embedding in static_embeddings.items():
                if vocab_word != last_word:
                    sim = np.dot(last_word_embedding, vocab_embedding)
                    similarities[vocab_word] = sim
            
            st.write(f"**2. Similarity Search:** It compares `{last_word}`'s vector to all other word vectors.")
            
            prediction = max(similarities, key=similarities.get)
            st.success(f"**3. Prediction:** The word with the most similar vector is **'{prediction}'**.")
        else:
            st.error(f"The word '{last_word}' is not in our vocabulary.")

    with st.expander("Show Static Embedding Prediction Code"):
        st.code("""
import numpy as np

def predict_with_static_embedding(vocab, sequence):
    # Simulate embeddings for the demo
    np.random.seed(42)
    static_embeddings = {word: np.random.rand(8) for word in vocab}
    
    last_word = sequence[-1]
    if last_word not in static_embeddings: return None

    last_word_embedding = static_embeddings[last_word]
    
    similarities = {}
    for vocab_word, vocab_embedding in static_embeddings.items():
        if vocab_word != last_word:
            sim = np.dot(last_word_embedding, vocab_embedding)
            similarities[vocab_word] = sim
            
    if not similarities: return None
    return max(similarities, key=similarities.get)
        """, language='python')

    # --- Interactive Demo 3: Attention-based Prediction ---
    st.markdown("---")
    st.subheader("3. Attention-based Prediction Workbench")
    st.markdown("This method creates a blended 'context vector' from the entire sequence, weighted by attention scores, and then finds the word most similar to that new vector.")

    attention_sequence = st.text_input("Enter a seed phrase for Attention:", "the cat sat on")
    if st.button("Predict with Attention"):
        # Simulate embeddings and matrices
        embedding_dim = 8
        np.random.seed(42)
        static_embeddings = {word: np.random.rand(embedding_dim) for word in vocab}
        Wq, Wk, Wv = [np.random.rand(embedding_dim, embedding_dim) for _ in range(3)]
        
        context_words = attention_sequence.lower().split()
        st.write(f"**1. Context:** The model sees the full sequence: `{context_words}`.")

        # Get Q, K, V
        qkv = {w: (np.dot(static_embeddings[w], Wq), np.dot(static_embeddings[w], Wk), np.dot(static_embeddings[w], Wv)) for w in context_words}
        queries, keys, values = [list(x) for x in zip(*[qkv[w] for w in context_words])]
        
        # Calculate attention
        scores = np.dot(queries[-1], np.array(keys).T)
        def softmax(x): return np.exp(x) / np.sum(np.exp(x))
        attention_weights = softmax(scores / np.sqrt(embedding_dim))
        st.write(f"**2. Attention Weights:** It calculates attention from the last word `'{context_words[-1]}'` to all other words:")
        st.bar_chart(pd.DataFrame(attention_weights, index=context_words))

        # Create context vector
        context_vector = np.dot(attention_weights, np.array(values))
        st.write("**3. Context Vector:** It creates a new vector by blending the 'Value' of all words according to the attention weights.")

        # Predict
        similarities = {}
        for vocab_word, vocab_embedding in static_embeddings.items():
            if vocab_word not in context_words:
                sim = np.dot(context_vector, vocab_embedding)
                similarities[vocab_word] = sim
        
        prediction = max(similarities, key=similarities.get)
        st.write("**4. Similarity Search:** It searches the vocabulary for the word most similar to this new blended context vector.")
        st.success(f"**5. Prediction:** The predicted word is **'{prediction}'**.")

    with st.expander("Show Attention-based Prediction Code"):
        st.code("""
# See the full, commented function in the expander in section 4.8 of the app.
# The code here would be a combination of the functions from that section.
        """, language='python')
