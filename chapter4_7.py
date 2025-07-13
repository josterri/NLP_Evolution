import streamlit as st
import numpy as np
import pandas as pd
import time
import re
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

def render_4_7():
    """Renders the cumulative exercise section."""
    st.subheader("4.7: Exercise - Predicting the Next 5 Words")
    st.markdown("""
    Let's put everything we've learned together in a final exercise. We will start with a paragraph of text and a short seed phrase, and then use the principles of embeddings and attention to predict the next five words, one at a time. This is a simplified simulation of how a modern language model works.
    """)

    # --- Step 1: Input Text ---
    st.markdown("---")
    st.subheader("Step 1: The Input")
    text = st.text_area("Provide a paragraph of text to serve as our 'knowledge base':",
                        "The solar system is a gravitationally bound system of the Sun and the objects that orbit it. "
                        "Of the objects that orbit the Sun directly, the largest are the eight planets. "
                        "The remainder are smaller objects, such as the five dwarf planets and small solar system bodies.",
                        height=150)
    
    seed_phrase = st.text_input("Enter a seed phrase from the text above to start the prediction:", "the objects that orbit the")

    # --- Step 2: Setup Models ---
    st.markdown("---")
    st.subheader("Step 2: Setup - Building Our Models")
    st.markdown("First, we create a vocabulary and all the necessary models from our text.")

    # Preprocessing
    tokens = text.lower().replace('.', '').replace(',', '').split()
    vocab = sorted(list(set(tokens)))
    
    # --- Model 1: N-grams (Chapter 1) ---
    def build_ngram_model(tokens, n=3):
        model = defaultdict(Counter)
        for i in range(len(tokens) - n + 1):
            context, target = tuple(tokens[i:i+n-1]), tokens[i+n-1]
            model[context][target] += 1
        return model
    
    ngram_model = build_ngram_model(tokens)

    # --- Model 2: Static Embeddings (Chapter 2) ---
    embedding_dim = 8
    np.random.seed(42)
    static_embeddings = {word: np.random.rand(embedding_dim) for word in vocab}

    # --- Model 3: Attention Components (Chapter 4) ---
    W_query = np.random.rand(embedding_dim, embedding_dim)
    W_key = np.random.rand(embedding_dim, embedding_dim)
    W_value = np.random.rand(embedding_dim, embedding_dim)

    with st.expander("Show Setup Details"):
        st.write(f"**Vocabulary Size:** {len(vocab)}")
        st.write(f"**N-gram Model Contexts:** {len(ngram_model)}")
        st.write(f"**Embedding Dimension:** {embedding_dim}")

    # --- Step 3: The Prediction Loop ---
    st.markdown("---")
    st.subheader("Step 3: Compare Prediction Methods")
    if st.button("Predict the Next 5 Words with All Techniques"):
        
        # --- Method 1: N-gram Prediction ---
        st.markdown("#### 1. N-gram Prediction (from Chapter 1)")
        with st.expander("See N-gram Prediction Details"):
            st.markdown("""
            **How it works:** This method looks at the last two words of the sequence (a bigram context) and finds the single most frequent word that followed that exact pair in the original text. It has no knowledge of semantics.
            """)
            ngram_sequence = seed_phrase.lower().split()
            for i in range(5):
                st.write(f"**Prediction #{i+1}**")
                context = tuple(ngram_sequence[-2:])
                st.write(f"Current context: `{context}`")
                if context in ngram_model:
                    predicted_word = ngram_model[context].most_common(1)[0][0]
                    st.write(f"In the original text, the most common word to follow `{context}` was **'{predicted_word}'**.")
                    ngram_sequence.append(predicted_word)
                else:
                    st.error(f"The context `{context}` was never seen in the original text. Prediction stops.")
                    ngram_sequence.append("[UNKNOWN_CONTEXT]")
                    break
        st.info(f"**Result:** {' '.join(ngram_sequence)}")
        st.caption("N-grams are simple and fast, but often get stuck in repetitive loops or fail when a context is new.")

        # --- Method 2: Static Embedding Prediction ---
        st.markdown("#### 2. Static Embedding Prediction (from Chapter 2)")
        with st.expander("See Static Embedding Prediction Details"):
            st.markdown(f"""
            **How it works:** This method has a very short memory; its **context size is only 1**, meaning it ignores all context except for the single last word. 
            The prediction is a "similarity contest" in the vector space. It takes the static vector for the last word and searches our entire vocabulary of **{len(vocab)} words** to find the word whose vector is most similar.
            """)
            static_sequence = seed_phrase.lower().split()
            for i in range(5):
                st.write(f"**Prediction #{i+1}**")
                last_word = static_sequence[-1]
                st.write(f"1. The context is just the last word: `{last_word}`")
                last_word_embedding = static_embeddings.get(last_word)
                if last_word_embedding is not None:
                    st.write(f"2. We retrieve the static embedding (vector) for `{last_word}`.")
                    similarities = {}
                    for vocab_word, vocab_embedding in static_embeddings.items():
                        if vocab_word not in static_sequence:
                            sim = np.dot(last_word_embedding, vocab_embedding)
                            similarities[vocab_word] = sim
                    
                    st.write(f"3. We calculate the dot product between `{last_word}`'s vector and every other vector in the vocabulary to get a similarity score.")
                    
                    top_3_candidates = sorted(similarities, key=similarities.get, reverse=True)[:3]
                    predicted_word = top_3_candidates[0]
                    
                    st.write(f"4. The words with the highest similarity scores are: `{top_3_candidates}`. We pick the best one.")
                    st.write(f"5. The predicted next word is **'{predicted_word}'**.")
                    static_sequence.append(predicted_word)
                else:
                    st.error(f"Word `{last_word}` not found. Prediction stops.")
                    static_sequence.append("[UNKNOWN_WORD]")
                    break
        st.info(f"**Result:** {' '.join(static_sequence)}")
        st.caption("This method follows semantic similarity, but because it only sees the last word, it lacks any understanding of grammar or the broader topic, leading to 'topic drift'.")

        # --- Method 3: Attention-based Prediction ---
        st.markdown("#### 3. Attention-based Prediction (from Chapter 4)")
        with st.expander("See Attention-based Prediction Details"):
            st.markdown("""
            **How it works:** This is the most advanced method. It first calculates attention scores from the last word to all other words in the current sequence. It then creates a blended 'context vector' based on these scores. Finally, it finds the word in the vocabulary whose embedding is most similar to this new, rich context vector.
            """)
            attention_sequence = seed_phrase.lower().split()
            
            def get_qkv(word):
                embedding = static_embeddings.get(word, np.zeros(embedding_dim))
                return (np.dot(embedding, W_query), np.dot(embedding, W_key), np.dot(embedding, W_value))

            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            for i in range(5):
                st.markdown(f"--- \n#### Prediction #{i+1}")
                context_words = attention_sequence
                
                # Use set for unique words for QKV creation, but preserve order for sequence
                qkv_data = {word: get_qkv(word) for word in set(context_words)}
                queries = [qkv_data[word][0] for word in context_words]
                keys = [qkv_data[word][1] for word in context_words]
                values = [qkv_data[word][2] for word in context_words]
                
                target_word = context_words[-1]
                query_target = queries[-1]
                
                scores = np.array([np.dot(query_target, k) for k in keys])
                attention_weights = softmax(scores)
                
                st.write(f"**1. Calculating Attention:** For the current word `'{target_word}'`, we calculate its attention to all words in the sequence `{' '.join(context_words)}`.")
                
                # --- New Heatmap Visualization ---
                fig, ax = plt.subplots()
                df = pd.DataFrame([attention_weights], columns=context_words, index=[f"Attention from '{target_word}'"])
                sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", ax=ax, cbar=False)
                st.pyplot(fig)
                st.caption(f"This heatmap shows which words the model 'focused on' to understand `'{target_word}'` in this context.")
                
                context_vector = np.dot(attention_weights, values)
                st.write("**2. Creating Context Vector:** The model blends the 'Value' of all words using the attention weights to create a single context-aware vector.")
                
                similarities = {}
                for vocab_word, vocab_embedding in static_embeddings.items():
                    if vocab_word not in attention_sequence:
                        sim = np.dot(context_vector, vocab_embedding) / (np.linalg.norm(context_vector) * np.linalg.norm(vocab_embedding))
                        similarities[vocab_word] = sim
                
                top_3_candidates = sorted(similarities, key=similarities.get, reverse=True)[:3]
                predicted_word = top_3_candidates[0]
                
                st.write(f"**3. Predicting:** The model searches the vocabulary for the word most similar to this new context vector. The top candidates are `{top_3_candidates}`.")
                st.success(f"**Predicted Next Word:** {predicted_word}")
                attention_sequence.append(predicted_word)
                
        st.info(f"**Result:** {' '.join(attention_sequence)}")
        st.caption("By blending information from the entire context, this method often produces more relevant (though still not perfect) predictions.")
