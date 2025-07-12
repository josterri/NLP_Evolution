import streamlit as st
import re
import pandas as pd
import numpy as np
from collections import Counter
# Add required imports for the demo
import PyPDF2
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# --- NLTK Stopwords Download ---
# This is a one-time download. Streamlit handles caching.
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK resources (stopwords)...")
    nltk.download('stopwords')
    nltk.download('punkt')

def render_2_6():
    """Renders the Word2Vec in Action demo section."""
    st.subheader("2.6: Demo - Word2Vec in Action")
    st.markdown("""
    Theory is great, but let's see these concepts in action. In this demo, we will perform the full pipeline:
    1.  **Load Data:** Read the text from a PDF document.
    2.  **Preprocess Text:** Clean and tokenize the text to prepare it for the model.
    3.  **Train Model:** Train our own Word2Vec model on the document's content.
    4.  **Explore & Visualize:** Explore the learned embeddings to find similar words and visualize the vector space.
    """)
    st.warning("You may need to install some libraries: `pip install PyPDF2 gensim scikit-learn nltk`")

    # --- 1. Load Data ---
    st.markdown("---")
    st.subheader("Step 1: Load a PDF")
    st.markdown("You can upload your own text-based PDF, or use the default `BIS_Speech.pdf` by leaving the uploader empty.")
    
    uploaded_file = st.file_uploader("Choose a PDF file (Optional)", type="pdf")
    
    pdf_file = None
    file_name = ""

    if uploaded_file is not None:
        pdf_file = uploaded_file
        file_name = uploaded_file.name
    else:
        st.info("No file uploaded. Attempting to load default `BIS_Speech.pdf` from the same folder.")
        try:
            if os.path.exists("BIS_Speech.pdf"):
                pdf_file = open("BIS_Speech.pdf", "rb")
                file_name = "BIS_Speech.pdf"
            else:
                st.error("Default file `BIS_Speech.pdf` not found. Please place it in the same folder as the app or upload a file.")
        except Exception as e:
            st.error(f"Could not open default file: {e}")

    if pdf_file:
        try:
            # --- Text Extraction ---
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            raw_text = ""
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
            
            st.success(f"Successfully extracted {len(raw_text.split())} words from '{file_name}'.")

            with st.expander("Show Raw Extracted Text"):
                st.text_area("", raw_text, height=200)

            # --- 2. Preprocess Text ---
            st.markdown("---")
            st.subheader("Step 2: Preprocess Text")
            st.markdown("""
            Raw text is messy. To train a meaningful model, we need to clean and standardize it. This process, called preprocessing, is one of the most critical steps in any NLP task.
            - **Lowercase:** We convert all text to lowercase so that "The" and "the" are treated as the same word.
            - **Remove Punctuation/Numbers:** We remove characters that aren't letters or spaces to simplify the vocabulary.
            - **Tokenization:** We split the continuous string of text into a list of individual words (tokens).
            - **Stopword Removal:** We remove common words like 'a', 'the', 'in', 'is', which don't carry much semantic weight. This helps the model focus on the more meaningful words.
            """)
            
            # Simple preprocessing
            cleaned_text = raw_text.lower()
            cleaned_text = re.sub(r'[^a-z\s]', '', cleaned_text) # Remove punctuation and numbers
            tokens = word_tokenize(cleaned_text)
            stop_words = set(stopwords.words('english'))
            processed_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            
            with st.expander("Show Cleaned Text (after stopword removal, etc.)"):
                st.text_area("", ' '.join(processed_tokens), height=200)

            # --- 3. Train Word2Vec Model ---
            st.markdown("---")
            st.subheader("Step 3: Train Word2Vec Model")
            st.markdown("""
            Now we feed our clean list of tokens into the Word2Vec model. The model will iterate over the text and learn a vector for each word based on its context.
            """)
            
            if st.button("Train Word2Vec Model"):
                with st.spinner("Training model... This may take a moment."):
                    sentences = [processed_tokens] 
                    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
                    model.train(sentences, total_examples=len(sentences), epochs=20)
                    st.session_state.w2v_model = model
                    st.success(f"Model trained successfully! Vocabulary size: {len(model.wv.index_to_key)} words.")

                    with st.expander("Show Learned Vocabulary"):
                        st.write(model.wv.index_to_key)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Close the file if it was opened from disk
            if uploaded_file is None and pdf_file:
                pdf_file.close()

    # --- 4. Explore & Visualize ---
    if 'w2v_model' in st.session_state:
        model = st.session_state.w2v_model
        st.markdown("---")
        st.subheader("Step 4: Explore the Model")

        st.markdown("##### Find Similar Words")
        word_to_explore = st.text_input("Enter a word from the document to find similar words:", value="policy")
        if word_to_explore in model.wv:
            similar_words = model.wv.most_similar(word_to_explore)
            df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
            st.dataframe(df)
        else:
            st.warning("Word not in model's vocabulary. Try another word.")

        st.markdown("##### Visualize the Vector Space")
        with st.spinner("Generating visualization..."):
            try:
                X = model.wv[model.wv.index_to_key]
                pca = PCA(n_components=2)
                result = pca.fit_transform(X)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.scatter(result[:, 0], result[:, 1])
                
                words = list(model.wv.index_to_key)
                for i, word in enumerate(words[:50]):
                    ax.annotate(word, xy=(result[i, 0], result[i, 1]))
                
                ax.set_title("2D PCA Visualization of Word Embeddings")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate visualization: {e}")
