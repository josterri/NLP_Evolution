import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Use fallback implementations to avoid Python 3.13 sklearn compatibility issues
from sklearn_fallbacks import PCA, TfidfVectorizer
from collections import Counter, defaultdict
import PyPDF2
import os

# --- NLTK Stopwords Download ---
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK resources (stopwords)...")
    nltk.download('stopwords')
    nltk.download('punkt')

class SimpleWord2Vec:
    """A simplified Word2Vec implementation for educational purposes."""
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.vocabulary = {}
        self.word_vectors = {}
        self.word_counts = Counter()
        
    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences."""
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
        
        # Filter words by minimum count
        filtered_words = [word for word, count in self.word_counts.items() 
                         if count >= self.min_count]
        
        # Create vocabulary mapping
        self.vocabulary = {word: i for i, word in enumerate(filtered_words)}
        
        # Initialize random word vectors
        vocab_size = len(self.vocabulary)
        self.word_vectors = {
            word: np.random.randn(self.vector_size) * 0.1
            for word in self.vocabulary
        }
        
        return self.vocabulary
    
    def get_context_pairs(self, sentence):
        """Get context pairs for training."""
        pairs = []
        for i, target_word in enumerate(sentence):
            if target_word not in self.vocabulary:
                continue
                
            # Get context window
            start = max(0, i - self.window)
            end = min(len(sentence), i + self.window + 1)
            
            for j in range(start, end):
                if i != j and sentence[j] in self.vocabulary:
                    pairs.append((target_word, sentence[j]))
        
        return pairs
    
    def train(self, sentences, epochs=20):
        """Simple training simulation."""
        st.info("Training Word2Vec model (simplified educational version)...")
        
        # Build vocabulary
        self.build_vocabulary(sentences)
        
        # Simulate training updates
        for epoch in range(epochs):
            for sentence in sentences:
                pairs = self.get_context_pairs(sentence)
                
                # Simulate gradient updates (simplified)
                for target, context in pairs:
                    if target in self.word_vectors and context in self.word_vectors:
                        # Simple update: make similar words more similar
                        target_vec = self.word_vectors[target]
                        context_vec = self.word_vectors[context]
                        
                        # Move vectors slightly closer (simplified update)
                        learning_rate = 0.01 * (1 - epoch / epochs)  # Decay learning rate
                        diff = context_vec - target_vec
                        self.word_vectors[target] += learning_rate * diff * 0.1
                        self.word_vectors[context] -= learning_rate * diff * 0.1
    
    def most_similar(self, word, topn=10):
        """Find most similar words using cosine similarity."""
        if word not in self.word_vectors:
            return []
        
        target_vec = self.word_vectors[word]
        similarities = []
        
        for other_word, other_vec in self.word_vectors.items():
            if other_word != word:
                # Cosine similarity
                similarity = np.dot(target_vec, other_vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    def get_vector(self, word):
        """Get vector for a word."""
        return self.word_vectors.get(word, np.zeros(self.vector_size))

def render_2_6():
    """Renders the Word2Vec in Action demo section."""
    st.subheader("2.6: Demo - Word2Vec in Action")
    st.markdown("""
    Theory is great, but let's see these concepts in action. In this demo, we will perform the full pipeline:
    1.  **Load Data:** Read the text from a PDF document.
    2.  **Preprocess Text:** Clean and tokenize the text to prepare it for the model.
    3.  **Train Model:** Train our own simplified Word2Vec model on the document's content.
    4.  **Explore & Visualize:** Explore the learned embeddings to find similar words and visualize the vector space.
    """)
    
    st.info("📝 Note: This uses a simplified educational version of Word2Vec for demonstration purposes.")

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

    # Alternative: Use sample text if no PDF
    if not pdf_file:
        st.info("Using sample text for demonstration...")
        raw_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and 
        human language, in particular how to program computers to process and analyze 
        large amounts of natural language data. The goal is a computer capable of 
        understanding the contents of documents, including the contextual nuances of 
        the language within them. The technology can then accurately extract information 
        and insights contained in the documents as well as categorize and organize the 
        documents themselves. Machine learning and deep learning have revolutionized 
        natural language processing in recent years. Word embeddings like Word2Vec 
        and GloVe have enabled computers to understand semantic relationships between 
        words. Transformer models like BERT and GPT have achieved remarkable results 
        in understanding and generating human language.
        """
        file_name = "Sample Text"
    
    if pdf_file or raw_text:
        try:
            if pdf_file:
                # --- Text Extraction ---
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                raw_text = ""
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            
            st.success(f"Successfully loaded {len(raw_text.split())} words from '{file_name}'.")

            with st.expander("Show Raw Extracted Text"):
                st.text_area("", raw_text, height=200)

            # --- 2. Preprocess Text ---
            st.markdown("---")
            st.subheader("Step 2: Preprocess Text")
            st.markdown("""
            Raw text is messy. To train a meaningful model, we need to clean and standardize it:
            - **Lowercase:** Convert all text to lowercase so "The" and "the" are treated the same
            - **Remove Punctuation/Numbers:** Remove non-letter characters to simplify vocabulary
            - **Tokenization:** Split text into individual words (tokens)
            - **Stopword Removal:** Remove common words like 'a', 'the', 'in' that don't carry much meaning
            """)
            
            # Simple preprocessing
            cleaned_text = raw_text.lower()
            cleaned_text = re.sub(r'[^a-z\s]', '', cleaned_text)
            tokens = word_tokenize(cleaned_text)
            stop_words = set(stopwords.words('english'))
            processed_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            
            with st.expander("Show Cleaned Text (after stopword removal, etc.)"):
                st.text_area("", ' '.join(processed_tokens), height=200)

            # --- 3. Train Word2Vec Model ---
            st.markdown("---")
            st.subheader("Step 3: Train Simplified Word2Vec Model")
            st.markdown("""
            Now we feed our clean tokens into our simplified Word2Vec model. The model will learn 
            vector representations for each word based on the words that appear near it.
            """)
            
            if st.button("Train Word2Vec Model"):
                with st.spinner("Training model... This may take a moment."):
                    sentences = [processed_tokens]  # Simple single sentence approach
                    model = SimpleWord2Vec(vector_size=50, window=5, min_count=2)
                    model.train(sentences, epochs=20)
                    st.session_state.w2v_model = model
                    st.success(f"Model trained successfully! Vocabulary size: {len(model.vocabulary)} words.")

                    with st.expander("Show Learned Vocabulary"):
                        vocab_words = list(model.vocabulary.keys())
                        st.write(f"Vocabulary ({len(vocab_words)} words):")
                        st.write(vocab_words)

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
        st.markdown("Enter a word to find semantically similar words based on the learned embeddings.")
        
        available_words = list(model.vocabulary.keys())
        if available_words:
            word_to_explore = st.selectbox("Choose a word from the vocabulary:", available_words)
            
            if st.button("Find Similar Words"):
                similar_words = model.most_similar(word_to_explore)
                if similar_words:
                    df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
                    st.dataframe(df)
                else:
                    st.warning("No similar words found.")
        else:
            st.warning("No vocabulary available. Please train the model first.")

        st.markdown("##### Visualize the Vector Space")
        if st.button("Generate Visualization"):
            with st.spinner("Generating visualization..."):
                try:
                    # Get all word vectors
                    words = list(model.word_vectors.keys())
                    vectors = [model.word_vectors[word] for word in words]
                    
                    if len(vectors) > 1:
                        # Apply PCA to reduce to 2D
                        pca = PCA(n_components=2)
                        result = pca.fit_transform(vectors)
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.scatter(result[:, 0], result[:, 1], alpha=0.6)
                        
                        # Add word labels (limit to avoid clutter)
                        for i, word in enumerate(words[:min(30, len(words))]):
                            ax.annotate(word, xy=(result[i, 0], result[i, 1]), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7)
                        
                        ax.set_title("2D PCA Visualization of Word Embeddings")
                        ax.set_xlabel("First Principal Component")
                        ax.set_ylabel("Second Principal Component")
                        st.pyplot(fig)
                        
                        st.info("""
                        **Understanding the Visualization:**
                        - Each point represents a word in the 2D projected space
                        - Words that appear in similar contexts should cluster together
                        - The distance between points reflects semantic similarity
                        """)
                    else:
                        st.warning("Need at least 2 words to create visualization.")
                        
                except Exception as e:
                    st.error(f"Could not generate visualization: {e}")

    # Educational notes
    st.markdown("---")
    st.markdown("### 📚 Educational Notes")
    st.info("""
    **What you've learned:**
    - How to preprocess text for machine learning
    - The basic concept of training word embeddings
    - How similar words cluster in vector space
    - The importance of context in determining word meaning
    
    **Real-world differences:**
    - Production Word2Vec models use negative sampling and hierarchical softmax
    - They're trained on much larger corpora (billions of words)
    - Real embeddings have higher dimensions (typically 300-1000)
    - Training takes hours/days on powerful hardware
    """)