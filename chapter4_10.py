import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict

def render_4_10():
    """Renders the long text generation exercise section."""
    st.subheader("4.10: Long Exercise - Create Your Own Predicted Text")
    st.markdown("""
    This final exercise for Chapter 4 is a complete, hands-on challenge. You will act as the architect of a simple text generation system based on the attention principles we've learned, and compare it to the previous methods.

    **Your Task:** Provide a body of text (your model's 'knowledge') and a starting phrase. Then, generate a continuation of that phrase using all three major techniques we've studied.
    """)

    # --- Step 1: The Corpus ---
    st.markdown("---")
    st.subheader("Step 1: Provide the Corpus")
    st.markdown("This text will be used to create the vocabulary and the static word embeddings that form the foundation of our model.")
    corpus = st.text_area("Enter your text corpus here:",
                          "The journey into space is a challenge. The rocket needs powerful engines to escape gravity. "
                          "Once in orbit, the spaceship follows a precise path. The crew must be brave. "
                          "This mission to the moon is a great challenge for the brave crew.",
                          height=150)

    # --- Step 2: The Seed ---
    st.markdown("---")
    st.subheader("Step 2: Provide the Seed Phrase")
    st.markdown("This is the starting point for your text generation.")
    seed_phrase = st.text_input("Enter a seed phrase to begin generation:", "the brave crew must")

    # --- Step 3: Generate ---
    st.markdown("---")
    st.subheader("Step 3: Generate and Compare")
    num_words_to_generate = st.slider("Number of words to generate:", 1, 10, 5)

    if st.button("Generate Text with All Methods"):
        with st.spinner("Building models and generating text..."):
            # --- Setup on the fly ---
            tokens = corpus.lower().replace('.', '').replace(',', '').split()
            vocab = sorted(list(set(tokens)))
            embedding_dim = 16 # A slightly larger dimension
            np.random.seed(sum(ord(c) for c in corpus)) # Seed based on corpus
            static_embeddings = {word: np.random.rand(embedding_dim) for word in vocab}
            Wq, Wk, Wv = [np.random.rand(embedding_dim, embedding_dim) for _ in range(3)]

            # --- Experiment 1: N-gram ---
            st.markdown("---")
            st.subheader("Experiment 1: N-gram Prediction")
            st.markdown("**Method:** Looks at the last 2 words and predicts the most frequent word that followed them in the corpus.")
            ngram_model = defaultdict(Counter)
            for i in range(len(tokens) - 3 + 1):
                context, target = tuple(tokens[i:i+2]), tokens[i+2]
                ngram_model[context][target] += 1
            
            ngram_sequence = seed_phrase.lower().split()
            for _ in range(num_words_to_generate):
                context = tuple(ngram_sequence[-2:])
                if context in ngram_model:
                    predicted_word = ngram_model[context].most_common(1)[0][0]
                    ngram_sequence.append(predicted_word)
                else:
                    ngram_sequence.append("[UNKNOWN]")
                    break
            st.info(f"**Generated Text:** {' '.join(ngram_sequence)}")
            st.caption("**Analysis:** This method is very literal. It can only reproduce sequences it has seen before, which often leads to repetitive or nonsensical results when a context is new.")

            # --- Experiment 2: Static Embedding ---
            st.markdown("---")
            st.subheader("Experiment 2: Static Embedding Prediction")
            st.markdown("**Method:** Looks at the *single last word*, finds its vector, and predicts the word with the most similar vector in the entire vocabulary.")
            static_sequence = seed_phrase.lower().split()
            for _ in range(num_words_to_generate):
                last_word = static_sequence[-1]
                if last_word in static_embeddings:
                    last_word_embedding = static_embeddings[last_word]
                    similarities = {word: np.dot(last_word_embedding, emb) for word, emb in static_embeddings.items() if word not in static_sequence}
                    if not similarities: break
                    predicted_word = max(similarities, key=similarities.get)
                    static_sequence.append(predicted_word)
                else:
                    static_sequence.append("[UNKNOWN]")
                    break
            st.info(f"**Generated Text:** {' '.join(static_sequence)}")
            st.caption("**Analysis:** This method follows a 'semantic chain' but has no memory of the sentence's topic. It can easily drift off-topic by following the similarities of the most recent word.")

            # --- Experiment 3: Attention-based ---
            st.markdown("---")
            st.subheader("Experiment 3: Attention-based Prediction")
            st.markdown("**Method:** Looks at the *entire sequence*, calculates attention from the last word to all others, creates a blended context vector, and then predicts the word most similar to that *new context vector*.")
            attention_sequence = seed_phrase.lower().split()
            
            def get_qkv(word):
                embedding = static_embeddings.get(word, np.zeros(embedding_dim))
                return (np.dot(embedding, Wq), np.dot(embedding, Wk), np.dot(embedding, Wv))

            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            for _ in range(num_words_to_generate):
                context_words = attention_sequence
                qkv_data = {word: get_qkv(word) for word in set(context_words)}
                queries = [qkv_data[word][0] for word in context_words]
                keys = [qkv_data[word][1] for word in context_words]
                values = [qkv_data[word][2] for word in context_words]
                
                query_target = queries[-1]
                scores = np.array([np.dot(query_target, k) for k in keys])
                attention_weights = softmax(scores / np.sqrt(embedding_dim))
                
                context_vector = np.dot(attention_weights, values)
                
                similarities = {}
                for vocab_word, vocab_embedding in static_embeddings.items():
                    if vocab_word not in attention_sequence:
                        sim = np.dot(context_vector, vocab_embedding) / (np.linalg.norm(context_vector) * np.linalg.norm(vocab_embedding))
                        similarities[vocab_word] = sim
                
                if not similarities: break
                predicted_word = max(similarities, key=similarities.get)
                attention_sequence.append(predicted_word)

            st.info(f"**Generated Text:** {' '.join(attention_sequence)}")
            st.caption("**Analysis:** By considering the full context, this method is better at staying on topic. The blended context vector provides a much richer signal for what the next word should be about.")

    # --- Step 4: Analysis ---
    st.markdown("---")
    st.subheader("Step 4: Analysis Questions")
    st.markdown("""
    1.  **Compare the Results:** Run the generator. Which of the three generated texts seems the most coherent or "on topic"? Which one is the most repetitive or nonsensical? Why?
    2.  **Change the Corpus:** Keep the seed phrase the same, but change the corpus to be about cooking (e.g., "The recipe needs fresh ingredients..."). How does the generated text change for each method?
    3.  **Limitations:** Does the attention-based text always make perfect grammatical sense? What key component is our simplified model still missing that a full Transformer model (Chapter 5) would have? (Hint: Think about grammar and sentence structure).
    """)


    # --- Step 5: Analysis Answers ---
    st.markdown("---")
    st.subheader("Step 5: Analysis Answers")
    with st.expander("Click to see the answers to the analysis questions"):
        st.markdown("""
        **1. Compare the Results:**
        -   **Most Coherent/On-Topic: Attention-based Prediction.** This method is the most coherent because it's the only one that considers the *entire* context of the sentence it's building. By creating a blended "context vector" from all the words in the sequence, it has a much better idea of the overall topic. The N-gram and Static Embedding methods have very short memories and get easily lost.
        -   **Most Repetitive: N-gram Prediction.** The N-gram model is based purely on frequency and has no understanding of meaning. If the most common word to follow the phrase `("the", "brave")` is `crew`, it will *always* predict `crew`. This makes it very easy for the model to get stuck in simple, common loops that it has seen in the training text.
        -   **Most Nonsensical (or "Drifting"): Static Embedding Prediction.** This method is most likely to drift off-topic because its memory is only one word long. It looks at the very last word and finds a semantically similar word. For example, if it generates `...brave crew mission`, its next prediction will be based only on words similar to "mission" (like "objective", "goal", "task"), completely forgetting the original context of "brave crew" or "space".

        ---
        
        **2. Change the Corpus:**
        
        If you change the corpus to be about cooking, all three methods will generate completely different text, because the models' entire "knowledge" comes from the text they are trained on. They have no outside knowledge of the world.
        -   **N-gram:** It would start generating common phrases from your cooking text, like `fresh ingredients need...`.
        -   **Static Embedding:** It would generate a chain of semantically related cooking words, for example: `...add fresh basil then green...` (since "basil" and "green" are semantically related).
        -   **Attention:** It would try to generate a more topically consistent cooking-related phrase by looking at the entire seed phrase.
        
        This demonstrates the most fundamental concept in machine learning: the model is only as good as the data it's trained on.

        ---

        **3. Limitations:**
        
        No, the attention-based text in our exercise will often not make perfect grammatical sense. While it's better at staying on topic, our simplified model is still missing several key components of a full Transformer (which we will cover in Chapter 5):
        1.  **Positional Encodings:** Our model has no inherent sense of word order. It treats `["the", "brave", "crew"]` and `["crew", "brave", "the"]` as the same "bag" of words when creating the context. A real Transformer adds a special "positional vector" to each word's embedding so the model knows which word came first, second, third, etc. This is absolutely critical for understanding grammar.
        2.  **Feed-Forward Layers:** After the attention calculation, a real Transformer passes the new, context-rich vector through another set of neural network layers. These layers help the model make a more intelligent final prediction based on the information gathered by the attention mechanism.
        3.  **Multiple Layers and Heads:** Our model has only one attention "head" and one layer. A full Transformer stacks many of these layers on top of each other, with multiple attention heads in each layer. This allows the model to learn many different types of relationships (subject-verb, adjective-noun, etc.) simultaneously, resulting in a much deeper understanding of the text.
        """)