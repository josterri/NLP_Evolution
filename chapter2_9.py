import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
# Use fallback implementation to avoid Python 3.13 sklearn compatibility issues
from sklearn_fallbacks import PCA
import matplotlib.pyplot as plt

# Pre-defined embeddings for demonstration (subset of actual GloVe vectors)
SAMPLE_EMBEDDINGS = {
    'king': [0.50451, 0.68607, -0.59517, 0.32457, 0.22923, -0.67897, 0.41873, 0.15662, 0.29885, 0.73959],
    'queen': [0.37854, 0.53241, -0.45629, 0.22567, 0.42891, -0.69574, 0.45862, 0.27834, 0.34765, 0.68723],
    'man': [0.42567, 0.76234, -0.62341, 0.15673, 0.28954, -0.58732, 0.34521, 0.19847, 0.41283, 0.71659],
    'woman': [0.34682, 0.59847, -0.48573, 0.31849, 0.35721, -0.62849, 0.42637, 0.25184, 0.38947, 0.66284],
    'dog': [0.12345, 0.45678, -0.23456, 0.67890, 0.34567, -0.78901, 0.56789, 0.12345, 0.67890, 0.23456],
    'cat': [0.23456, 0.56789, -0.34567, 0.78901, 0.45678, -0.67890, 0.67890, 0.23456, 0.78901, 0.34567],
    'happy': [0.65432, 0.32109, -0.87654, 0.54321, 0.21098, -0.76543, 0.43210, 0.65432, 0.32109, 0.87654],
    'sad': [0.54321, 0.21098, -0.76543, 0.43210, 0.65432, -0.87654, 0.32109, 0.54321, 0.21098, 0.76543],
    'good': [0.76543, 0.43210, -0.65432, 0.32109, 0.87654, -0.54321, 0.21098, 0.76543, 0.43210, 0.65432],
    'bad': [0.43210, 0.65432, -0.32109, 0.87654, 0.54321, -0.21098, 0.76543, 0.43210, 0.65432, 0.32109],
    'computer': [0.87654, 0.54321, -0.21098, 0.76543, 0.43210, -0.65432, 0.32109, 0.87654, 0.54321, 0.21098],
    'technology': [0.78901, 0.45678, -0.23456, 0.67890, 0.34567, -0.56789, 0.12345, 0.78901, 0.45678, 0.23456],
    'science': [0.67890, 0.34567, -0.56789, 0.12345, 0.78901, -0.23456, 0.45678, 0.67890, 0.34567, 0.56789],
    'math': [0.56789, 0.23456, -0.78901, 0.45678, 0.67890, -0.34567, 0.12345, 0.56789, 0.23456, 0.78901],
    'book': [0.45678, 0.67890, -0.34567, 0.78901, 0.23456, -0.56789, 0.12345, 0.45678, 0.67890, 0.34567],
    'read': [0.34567, 0.78901, -0.45678, 0.23456, 0.56789, -0.67890, 0.12345, 0.34567, 0.78901, 0.45678],
    'write': [0.23456, 0.56789, -0.67890, 0.12345, 0.45678, -0.78901, 0.34567, 0.23456, 0.56789, 0.67890],
    'car': [0.12345, 0.78901, -0.56789, 0.34567, 0.67890, -0.23456, 0.45678, 0.12345, 0.78901, 0.56789],
    'house': [0.21098, 0.87654, -0.43210, 0.65432, 0.32109, -0.76543, 0.54321, 0.21098, 0.87654, 0.43210],
    'city': [0.32109, 0.76543, -0.54321, 0.21098, 0.87654, -0.43210, 0.65432, 0.32109, 0.76543, 0.54321],
    'music': [0.43210, 0.65432, -0.87654, 0.32109, 0.76543, -0.54321, 0.21098, 0.43210, 0.65432, 0.87654],
    'art': [0.54321, 0.21098, -0.76543, 0.43210, 0.65432, -0.87654, 0.32109, 0.54321, 0.21098, 0.76543],
    'love': [0.65432, 0.32109, -0.87654, 0.54321, 0.21098, -0.76543, 0.43210, 0.65432, 0.32109, 0.87654],
    'hate': [0.76543, 0.43210, -0.65432, 0.32109, 0.87654, -0.54321, 0.21098, 0.76543, 0.43210, 0.65432],
    'water': [0.87654, 0.54321, -0.21098, 0.76543, 0.43210, -0.65432, 0.32109, 0.87654, 0.54321, 0.21098],
    'fire': [0.78901, 0.45678, -0.23456, 0.67890, 0.34567, -0.56789, 0.12345, 0.78901, 0.45678, 0.23456]
}

class SimpleEmbeddingModel:
    """A simplified embedding model using pre-defined vectors."""
    
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or SAMPLE_EMBEDDINGS
        self.vector_size = len(next(iter(self.embeddings.values())))
    
    def get_vector(self, word):
        """Get vector for a word."""
        return np.array(self.embeddings.get(word.lower(), np.zeros(self.vector_size)))
    
    def most_similar(self, word, topn=10):
        """Find most similar words using cosine similarity."""
        word = word.lower()
        if word not in self.embeddings:
            return []
        
        target_vec = np.array(self.embeddings[word])
        similarities = []
        
        for other_word, other_vec in self.embeddings.items():
            if other_word != word:
                other_vec = np.array(other_vec)
                # Cosine similarity
                similarity = np.dot(target_vec, other_vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    def analogy(self, positive, negative):
        """Solve analogies: positive[0] - negative[0] + positive[1] ≈ ?"""
        if len(positive) != 2 or len(negative) != 1:
            return []
        
        word1, word2 = [w.lower() for w in positive]
        word3 = negative[0].lower()
        
        if not all(w in self.embeddings for w in [word1, word2, word3]):
            return []
        
        # Calculate: word1 - word3 + word2
        vec1 = np.array(self.embeddings[word1])
        vec2 = np.array(self.embeddings[word2])
        vec3 = np.array(self.embeddings[word3])
        
        result_vec = vec1 - vec3 + vec2
        
        # Find most similar words to the result
        similarities = []
        for word, vec in self.embeddings.items():
            if word not in [word1, word2, word3]:
                vec = np.array(vec)
                similarity = np.dot(result_vec, vec) / (
                    np.linalg.norm(result_vec) * np.linalg.norm(vec)
                )
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:10]

def render_2_9():
    """Renders the pre-trained model demo section."""
    st.subheader("2.9: The Power of Pre-trained Models")
    st.markdown("""
    Training a Word2Vec model from scratch is great for understanding the process, but the results 
    are limited by the size and diversity of our training text. To get truly powerful and nuanced 
    embeddings, we can use **pre-trained models**.

    These are models that have already been trained by researchers on massive, high-quality text 
    corpora (like all of Wikipedia or Google News), containing billions of words. Using them saves 
    us immense time and computational resources, and gives us access to a much richer semantic space.
    """)
    
    st.info("""
    📝 **Note:** This demo uses a simplified subset of pre-trained embeddings for educational purposes. 
    Real pre-trained models like GloVe contain hundreds of thousands of words with much richer representations.
    """)
    
    # Initialize model
    model = SimpleEmbeddingModel()
    
    st.success("Sample pre-trained embeddings loaded successfully!")

    st.markdown("---")
    st.subheader("Explore the Pre-trained Model")

    # Show available vocabulary
    with st.expander("Show Available Vocabulary"):
        vocab_words = list(model.embeddings.keys())
        st.write(f"**Sample vocabulary ({len(vocab_words)} words):**")
        cols = st.columns(4)
        for i, word in enumerate(vocab_words):
            cols[i % 4].write(f"• {word}")

    # --- Find Similar Words ---
    st.markdown("##### Find Similar Words")
    st.markdown("With a model trained on a huge corpus, the concept of 'similarity' becomes much more robust and interesting.")
    
    word_to_explore = st.selectbox(
        "Select a word to find similar words:",
        list(model.embeddings.keys()),
        index=list(model.embeddings.keys()).index('king')
    )
    
    if st.button("Find Similar Words"):
        similar_words = model.most_similar(word_to_explore)
        if similar_words:
            df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
            st.dataframe(df)
            
            # Visualize similarities
            fig = go.Figure(data=[
                go.Bar(x=[w[0] for w in similar_words[:5]], 
                       y=[w[1] for w in similar_words[:5]],
                       text=[f"{w[1]:.3f}" for w in similar_words[:5]],
                       textposition='auto')
            ])
            fig.update_layout(
                title=f"Top 5 Words Similar to '{word_to_explore}'",
                xaxis_title="Word",
                yaxis_title="Similarity Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No similar words found.")

    # --- Solve Analogies ---
    st.markdown("---")
    st.markdown("##### Solve Analogies")
    st.markdown("The famous `king - man + woman ≈ queen` analogy works much better with a model that has a deep understanding of these concepts.")
    
    # Preset analogies
    preset_analogies = {
        "King - Man + Woman": (["king", "woman"], ["man"]),
        "Good - Bad + Happy": (["good", "happy"], ["bad"]),
        "Computer - Technology + Science": (["computer", "science"], ["technology"]),
        "Love - Happy + Sad": (["love", "sad"], ["happy"]),
        "Car - City + House": (["car", "house"], ["city"])
    }
    
    analogy_choice = st.selectbox("Choose a preset analogy:", list(preset_analogies.keys()))
    
    if analogy_choice:
        positive, negative = preset_analogies[analogy_choice]
        
        col1, col2, col3 = st.columns(3)
        p1 = col1.selectbox("Positive 1", list(model.embeddings.keys()), 
                           index=list(model.embeddings.keys()).index(positive[0]))
        n1 = col2.selectbox("Negative 1", list(model.embeddings.keys()), 
                           index=list(model.embeddings.keys()).index(negative[0]))
        p2 = col3.selectbox("Positive 2", list(model.embeddings.keys()), 
                           index=list(model.embeddings.keys()).index(positive[1]))

        if st.button("Solve Analogy"):
            result = model.analogy([p1, p2], [n1])
            if result:
                st.write(f"**Result for `{p1} - {n1} + {p2}`:**")
                df_analogy = pd.DataFrame(result[:5], columns=['Resulting Word', 'Similarity'])
                st.dataframe(df_analogy)
                
                # Show the math
                st.markdown("**Vector Math Explanation:**")
                st.latex(f"\\vec{{{p1}}} - \\vec{{{n1}}} + \\vec{{{p2}}} \\approx \\vec{{{result[0][0]}}}")
                
                # Visualize the analogy
                fig = go.Figure(data=[
                    go.Bar(x=[r[0] for r in result[:5]], 
                           y=[r[1] for r in result[:5]],
                           text=[f"{r[1]:.3f}" for r in result[:5]],
                           textposition='auto')
                ])
                fig.update_layout(
                    title=f"Analogy Results: {p1} - {n1} + {p2}",
                    xaxis_title="Word",
                    yaxis_title="Similarity Score"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not solve analogy with available words.")

    # --- Visualize Embeddings ---
    st.markdown("---")
    st.markdown("##### Visualize Word Embeddings")
    st.markdown("Let's visualize how these words are positioned in the embedding space.")
    
    words_to_plot = st.multiselect(
        "Select words to visualize:",
        list(model.embeddings.keys()),
        default=['king', 'queen', 'man', 'woman', 'dog', 'cat', 'happy', 'sad']
    )
    
    if len(words_to_plot) >= 2 and st.button("Generate Visualization"):
        with st.spinner("Generating visualization..."):
            # Get vectors for selected words
            vectors = [model.get_vector(word) for word in words_to_plot]
            
            # Apply PCA to reduce to 2D
            pca = PCA(n_components=2)
            result = pca.fit_transform(vectors)
            
            # Create interactive plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=result[:, 0],
                y=result[:, 1],
                mode='markers+text',
                text=words_to_plot,
                textposition='top center',
                textfont_size=12,
                marker=dict(size=10, color='blue', opacity=0.7),
                name='Words'
            ))
            
            fig.update_layout(
                title="2D Visualization of Word Embeddings (PCA)",
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component",
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.info("""
            **Understanding the Visualization:**
            - Each point represents a word in the 2D projected space
            - Words with similar meanings should cluster together
            - The distance between points reflects semantic similarity
            - PCA preserves the most important variations in the data
            """)

    # --- Educational Notes ---
    st.markdown("---")
    st.markdown("### 📚 Real Pre-trained Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Popular Pre-trained Models:**")
        st.markdown("""
        - **GloVe (Global Vectors)**: Trained on 6B tokens, 400K vocab
        - **Word2Vec Google News**: Trained on 100B words
        - **FastText**: Handles out-of-vocabulary words
        - **BERT Embeddings**: Contextual embeddings
        """)
    
    with col2:
        st.markdown("**Advantages of Pre-trained Models:**")
        st.markdown("""
        - ✅ Rich semantic understanding
        - ✅ Large vocabulary coverage
        - ✅ Ready to use immediately
        - ✅ Proven performance on many tasks
        - ✅ Save computational resources
        """)
    
    st.markdown("### 🔗 Getting Real Pre-trained Embeddings")
    st.info("""
    **For production use, you can download:**
    
    1. **GloVe**: https://nlp.stanford.edu/projects/glove/
    2. **Word2Vec**: https://code.google.com/archive/p/word2vec/
    3. **FastText**: https://fasttext.cc/docs/en/pretrained-vectors.html
    4. **Hugging Face Hub**: https://huggingface.co/models
    
    These models contain hundreds of thousands of words with much richer representations than our demo.
    """)

    # --- Comparison ---
    st.markdown("### 📊 From Scratch vs Pre-trained")
    
    comparison_data = {
        "Aspect": ["Vocabulary Size", "Training Time", "Quality", "Customization", "Resource Requirements"],
        "From Scratch": ["Small (100-1000)", "Minutes-Hours", "Limited", "High", "Low"],
        "Pre-trained": ["Large (100K-1M)", "Days-Weeks (already done)", "High", "Low", "High (for training)"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)
    
    st.success("""
    🎯 **Key Takeaway**: Pre-trained embeddings give you a massive head start by providing rich, 
    high-quality word representations learned from billions of words. They're the foundation for 
    most modern NLP applications.
    """)