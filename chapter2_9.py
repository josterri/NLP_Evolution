import streamlit as st
import gensim
import pandas as pd
import os
import numpy as np
import random

# --- Model Loading ---
# Use Streamlit's caching to load the model from disk only once.
# def load_model_from_disk(glove_file="glove.6B.100d.txt"):
@st.cache_resource
def load_model_from_disk(glove_file="glove.6B.50dsmall.txt"):
    """
    Loads a GloVe model from a text file on disk.
    If the file is not in word2vec format, it converts it.
    """
    word2vec_output_file = glove_file + '.word2vec'

    # --- Step 1: Provide download instructions if the file is missing ---
    if not os.path.exists(glove_file):
        st.error(f"Model file not found: `{glove_file}`")
        st.markdown("""
        **To use this demo, please do the following:**
        1.  Download the GloVe pre-trained vectors. A good starting point is `glove.6B.zip` (822 MB).
            You can download it from the official Stanford page: [**GloVe: Global Vectors for Word Representation**](https://nlp.stanford.edu/projects/glove/).
        2.  Unzip the file.
        3.  Place the `glove.6B.100d.txt` file in the same folder as this Streamlit app.
        4.  Rerun the app.
        """)
        st.stop()

    # --- Step 2: Convert GloVe format to Word2Vec format if necessary ---
    # This is a one-time conversion that makes loading faster next time.
    if not os.path.exists(word2vec_output_file):
        st.info(f"First-time setup: Converting `{glove_file}` to Word2Vec format...")
        st.info("This may take a minute but will only happen once.")
        try:
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(glove_file, word2vec_output_file)
            st.success("Conversion complete.")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
            st.stop()




    # --- Step 3: Load the model from the converted file ---
    st.info(f"Loading model `{word2vec_output_file}` from disk...")
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return model

def render_2_9():
    """Renders the pre-trained model demo section."""
    st.subheader("2.9: The Power of Pre-trained Models")
    st.markdown("""
    Training a Word2Vec model from scratch is great for understanding the process, but the results are limited by the size and diversity of our training text. To get truly powerful and nuanced embeddings, we can use **pre-trained models**.

    These are models that have already been trained by researchers on massive, high-quality text corpora (like all of Wikipedia or Google News), containing billions of words. Using them saves us immense time and computational resources, and gives us access to a much richer semantic space.
    """)
    
    try:
        model = load_model_from_disk()
        st.success("Pre-trained GloVe model loaded successfully from local disk!")

        st.markdown("---")
        st.subheader("Explore the Pre-trained Model")

        # --- Find Similar Words ---
        st.markdown("##### Find Similar Words")
        st.markdown("With a model trained on a huge corpus, the concept of 'similarity' becomes much more robust and interesting.")
        word_to_explore = st.text_input("Enter a word to find similar words:", value="carbon-based")
        
        try:
            if word_to_explore:
                similar_words = model.most_similar(word_to_explore)
                df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
                st.dataframe(df)
        except KeyError:
            st.error(f"Word '{word_to_explore}' not in the model's vocabulary. Try another common English word.")

        # --- Solve Analogies ---
        st.markdown("---")
        st.markdown("##### Solve Analogies")
        st.markdown("The famous `king - man + woman ≈ queen` analogy works much better with a model that has a deep understanding of these concepts.")
        
        c1, c2, c3 = st.columns(3)
        p1 = c1.text_input("Positive 1", "carbon-based")
        n1 = c2.text_input("Negative 1", "petroleum-based")
        p2 = c3.text_input("Positive 2", "superstrings")

        if st.button("Solve Analogy"):
            try:
                result = model.most_similar(positive=[p1, p2], negative=[n1], topn=5)
                df_analogy = pd.DataFrame(result, columns=['Resulting Word', 'Similarity'])
                st.write(f"Result for `{p1} - {n1} + {p2}`:")
                st.dataframe(df_analogy)
            except KeyError as e:
                st.error(f"One of the words is not in the vocabulary: {e}")

        # --- Generate a Sequence ---
        st.markdown("---")
        st.subheader("Generate a Sequence")
        st.markdown("We can use the rich semantic space of the model to extend a sentence. By finding words similar to the last word in a sequence, we can create a chain of related concepts.")
        
        seed_text = st.text_input("Enter a seed sentence (at least 5 words):", value="carbon-based is not petroleum-based")
        num_words_to_generate = st.slider("Number of additional words to generate:", 5, 50, 15)

        if st.button("Extend Sentence"):
            with st.spinner("Generating..."):
                words = seed_text.lower().split()
                if len(words) < 1:
                    st.error("Please provide at least one word in the seed sentence.")
                else:
                    for _ in range(num_words_to_generate):
                        try:
                            last_word = words[-1]
                            # Find similar words and pick one to add variety
                            similar = model.most_similar(last_word, topn=5)
                            next_word = random.choice([word for word, sim in similar])
                            words.append(next_word)
                        except (KeyError, IndexError):
                            # Stop if a word is not in vocab or no similar words are found
                            break
                    st.markdown("#### Generated Text:")
                    st.info(" ".join(words))
                    
                    st.warning("""
                    **Why isn't this a coherent sentence?**
                    You'll notice the generated text often drifts into strange loops or doesn't make grammatical sense. This is because our simple method has major limitations:
                    -   **No Memory:** It only ever looks at the *single last word*. It has no understanding of the overall topic, grammar, or sentence structure established by the earlier words.
                    -   **No Grammar Rules:** The model only knows that words are semantically similar (e.g., 'king' and 'queen' are related), but it has no built-in knowledge of how to form a proper sentence.
                    
                    This demonstrates why static embeddings alone are not enough for language generation. We need more advanced architectures like RNNs and Transformers (which we'll see in later chapters) to handle sequence and context properly.
                    """)


    except Exception as e:
        # This will catch the st.stop() from the loading function
        # as well as any other errors.
        pass

    st.markdown("---")
    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Country-Capital:** Try the analogy `france - paris + tokyo`. What do you get? What does this tell you about the relationships the model has learned?
    2.  **Verb Tense:** What about `walking - walk + swim`?
    3.  **Generate a Story:** Start a sentence generation with "the spaceship landed on the". What kind of story does the model create?
    """)

    st.subheader("🐍 The Python Behind Pre-trained Models")
    with st.expander("Show the Python Code for Using Pre-trained Models"):
        st.code("""
import streamlit as st
import gensim
import os
import random

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model_from_disk(glove_file="glove.6B.100d.txt"):
    # Define the output format for faster loading next time
    word2vec_output_file = glove_file + '.word2vec'

    # Check if the GloVe file exists. If not, stop.
    if not os.path.exists(glove_file):
        st.error(f"Model file not found: {glove_file}")
        st.stop()

    # Convert the GloVe file to Word2Vec format if it hasn't been done yet.
    # This is a one-time process.
    if not os.path.exists(word2vec_output_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_file, word2vec_output_file)

    # Load the converted model from disk
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return model

# --- Example Usage ---
# model = load_model_from_disk()

# Find similar words
# similar_to_computer = model.most_similar("computer")
# print(similar_to_computer)

# Solve an analogy
# result = model.most_similar(positive=['woman', 'king'], negative=['man'])
# print(result[0][0]) # queen

# Generate a sequence
def extend_sentence(model, seed_words, num_to_add):
    words = list(seed_words)
    for _ in range(num_to_add):
        try:
            last_word = words[-1]
            similar = model.most_similar(last_word, topn=5)
            next_word = random.choice([w for w, s in similar])
            words.append(next_word)
        except (KeyError, IndexError):
            break
    return " ".join(words)

# seed = ["the", "doctor", "went", "to", "the"]
# new_sentence = extend_sentence(model, seed, 10)
# print(new_sentence)
        """, language='python')
