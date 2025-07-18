"""
Comprehensive NLP Glossary for the Evolution of NLP app.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List

# Comprehensive NLP terms and definitions
NLP_TERMS = {
    "Attention Mechanism": {
        "definition": "A technique that allows neural networks to focus on specific parts of the input when producing an output, mimicking human attention.",
        "category": "Deep Learning",
        "chapter": "Chapter 4",
        "example": "In machine translation, attention helps the model focus on relevant source words when generating each target word."
    },
    "BERT": {
        "definition": "Bidirectional Encoder Representations from Transformers. A pre-trained transformer model that reads text bidirectionally.",
        "category": "Models",
        "chapter": "Chapter 8",
        "example": "BERT revolutionized NLP by learning context from both left and right sides of a word simultaneously."
    },
    "Backpropagation": {
        "definition": "An algorithm for computing gradients in neural networks by propagating errors backward through the network layers.",
        "category": "Machine Learning",
        "chapter": "Chapter 2",
        "example": "Backpropagation allows neural networks to learn by adjusting weights based on prediction errors."
    },
    "BLEU Score": {
        "definition": "Bilingual Evaluation Understudy - a metric for evaluating machine translation quality by comparing n-gram overlap with reference translations.",
        "category": "Evaluation",
        "chapter": "Chapter 6",
        "example": "A BLEU score of 30+ is generally considered good for machine translation systems."
    },
    "Byte Pair Encoding (BPE)": {
        "definition": "A subword tokenization method that iteratively merges the most frequent pairs of characters or character sequences.",
        "category": "Preprocessing",
        "chapter": "Chapter 4",
        "example": "BPE helps handle out-of-vocabulary words by breaking them into smaller, known subword units."
    },
    "Context Window": {
        "definition": "The number of surrounding words or tokens that a model considers when processing a particular position in the sequence.",
        "category": "Architecture",
        "chapter": "Chapter 1",
        "example": "A bigram model has a context window of 1, considering only the previous word."
    },
    "Cosine Similarity": {
        "definition": "A measure of similarity between two vectors based on the cosine of the angle between them, commonly used for word embeddings.",
        "category": "Mathematics",
        "chapter": "Chapter 2",
        "example": "Words with similar meanings have high cosine similarity in embedding space."
    },
    "Cross-Entropy Loss": {
        "definition": "A loss function commonly used in classification tasks that measures the difference between predicted and actual probability distributions.",
        "category": "Machine Learning",
        "chapter": "Chapter 2",
        "example": "Language models use cross-entropy loss to learn to predict the next word in a sequence."
    },
    "Decoder": {
        "definition": "In sequence-to-sequence models, the component that generates the output sequence from the encoded representation.",
        "category": "Architecture",
        "chapter": "Chapter 3",
        "example": "In machine translation, the decoder generates the target language sentence from the encoded source sentence."
    },
    "Embedding": {
        "definition": "A dense vector representation of words, sentences, or documents that captures semantic meaning in a continuous space.",
        "category": "Representation",
        "chapter": "Chapter 2",
        "example": "Word embeddings map words like 'king' and 'queen' to nearby points in vector space."
    },
    "Encoder": {
        "definition": "In sequence-to-sequence models, the component that processes the input sequence and creates a fixed-size representation.",
        "category": "Architecture",
        "chapter": "Chapter 3",
        "example": "The encoder in a translation model reads the source sentence and creates a context vector."
    },
    "Fine-tuning": {
        "definition": "The process of adapting a pre-trained model to a specific task by training it on task-specific data with a lower learning rate.",
        "category": "Training",
        "chapter": "Chapter 8",
        "example": "BERT can be fine-tuned for sentiment analysis by adding a classification layer and training on labeled data."
    },
    "GPT": {
        "definition": "Generative Pre-trained Transformer. A family of autoregressive language models trained to predict the next token in a sequence.",
        "category": "Models",
        "chapter": "Chapter 8",
        "example": "GPT models generate human-like text by predicting one word at a time based on previous context."
    },
    "Gradient Descent": {
        "definition": "An optimization algorithm that iteratively adjusts model parameters in the direction of steepest decrease of the loss function.",
        "category": "Optimization",
        "chapter": "Chapter 2",
        "example": "Neural networks use gradient descent to find parameter values that minimize prediction errors."
    },
    "Hidden State": {
        "definition": "In RNNs and similar architectures, the internal representation that carries information from previous time steps.",
        "category": "Architecture",
        "chapter": "Chapter 3",
        "example": "LSTM cells maintain hidden states to remember important information across long sequences."
    },
    "Hyperparameter": {
        "definition": "Configuration settings for machine learning algorithms that are set before training begins, such as learning rate or batch size.",
        "category": "Training",
        "chapter": "Chapter 2",
        "example": "The learning rate is a crucial hyperparameter that controls how quickly a model learns."
    },
    "Language Model": {
        "definition": "A statistical model that assigns probabilities to sequences of words, often used to predict the next word in a sequence.",
        "category": "Models",
        "chapter": "Chapter 1",
        "example": "N-gram models are simple language models that predict words based on previous n-1 words."
    },
    "LSTM": {
        "definition": "Long Short-Term Memory. A type of RNN designed to handle long sequences by using gates to control information flow.",
        "category": "Architecture",
        "chapter": "Chapter 3",
        "example": "LSTMs can remember information from early in a sentence when processing words at the end."
    },
    "Masked Language Model": {
        "definition": "A training objective where some tokens in the input are masked, and the model learns to predict the masked tokens.",
        "category": "Training",
        "chapter": "Chapter 8",
        "example": "BERT uses masked language modeling to learn bidirectional representations."
    },
    "Multi-Head Attention": {
        "definition": "An attention mechanism that performs multiple attention operations in parallel, each focusing on different types of relationships.",
        "category": "Architecture",
        "chapter": "Chapter 4",
        "example": "Transformers use multi-head attention to capture different linguistic relationships simultaneously."
    },
    "N-gram": {
        "definition": "A contiguous sequence of n items (usually words) from a text. Used in statistical language modeling.",
        "category": "Statistical NLP",
        "chapter": "Chapter 1",
        "example": "In the sentence 'the cat sat', the bigrams are 'the cat' and 'cat sat'."
    },
    "Named Entity Recognition (NER)": {
        "definition": "The task of identifying and classifying named entities (people, places, organizations) in text.",
        "category": "Tasks",
        "chapter": "Chapter 5",
        "example": "NER would identify 'Apple' as an organization and 'California' as a location."
    },
    "Neural Network": {
        "definition": "A computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information.",
        "category": "Machine Learning",
        "chapter": "Chapter 2",
        "example": "Feed-forward neural networks can learn complex patterns by combining simple mathematical operations."
    },
    "One-Hot Encoding": {
        "definition": "A representation where each word is represented as a vector with all zeros except for a single 1 at the word's index position.",
        "category": "Representation",
        "chapter": "Chapter 2",
        "example": "In a vocabulary of 1000 words, each word becomes a 1000-dimensional vector with one 1 and 999 zeros."
    },
    "Overfitting": {
        "definition": "When a model learns the training data too well, including noise, leading to poor performance on new, unseen data.",
        "category": "Machine Learning",
        "chapter": "Chapter 2",
        "example": "A language model that memorizes training sentences but can't generate coherent new text is overfitting."
    },
    "Part-of-Speech (POS) Tagging": {
        "definition": "The task of assigning grammatical categories (noun, verb, adjective, etc.) to words in a sentence.",
        "category": "Tasks",
        "chapter": "Chapter 0",
        "example": "In 'The cat runs', POS tagging assigns determiner, noun, and verb respectively."
    },
    "Perplexity": {
        "definition": "A measure of how well a language model predicts a sequence. Lower perplexity indicates better performance.",
        "category": "Evaluation",
        "chapter": "Chapter 1",
        "example": "A language model with perplexity 50 is more uncertain about predictions than one with perplexity 20."
    },
    "Positional Encoding": {
        "definition": "A method to inject information about token positions into transformer models, which don't inherently understand sequence order.",
        "category": "Architecture",
        "chapter": "Chapter 4",
        "example": "Transformers add positional encodings to embeddings so the model knows word order."
    },
    "Pre-training": {
        "definition": "Training a model on a large, general dataset before fine-tuning it for specific tasks.",
        "category": "Training",
        "chapter": "Chapter 8",
        "example": "GPT models are pre-trained on vast amounts of internet text before being adapted for specific applications."
    },
    "Prompt Engineering": {
        "definition": "The practice of designing input prompts to elicit desired outputs from language models.",
        "category": "Applications",
        "chapter": "Chapter 8",
        "example": "Adding 'Think step by step:' to a prompt can improve a model's reasoning performance."
    },
    "Recurrent Neural Network (RNN)": {
        "definition": "A neural network architecture designed for sequential data, with connections that create loops allowing information to persist.",
        "category": "Architecture",
        "chapter": "Chapter 3",
        "example": "RNNs process sentences word by word, maintaining a hidden state that captures context."
    },
    "ROUGE": {
        "definition": "Recall-Oriented Understudy for Gisting Evaluation. A set of metrics for evaluating automatic summarization and machine translation.",
        "category": "Evaluation",
        "chapter": "Chapter 6",
        "example": "ROUGE-L measures the longest common subsequence between generated and reference summaries."
    },
    "Self-Attention": {
        "definition": "An attention mechanism where a sequence attends to itself, allowing each position to relate to all positions in the same sequence.",
        "category": "Architecture",
        "chapter": "Chapter 4",
        "example": "In self-attention, the word 'it' can attend to earlier words in the sentence to determine what 'it' refers to."
    },
    "Semantic Similarity": {
        "definition": "A measure of how similar two pieces of text are in terms of meaning, regardless of exact word matches.",
        "category": "Semantics",
        "chapter": "Chapter 2",
        "example": "'Car accident' and 'vehicle crash' have high semantic similarity despite different words."
    },
    "Sentiment Analysis": {
        "definition": "The task of determining the emotional tone or opinion expressed in a piece of text (positive, negative, neutral).",
        "category": "Tasks",
        "chapter": "Chapter 5",
        "example": "Sentiment analysis would classify 'I love this movie!' as positive sentiment."
    },
    "Sequence-to-Sequence (Seq2Seq)": {
        "definition": "A neural network architecture that transforms one sequence into another, commonly used for translation and summarization.",
        "category": "Architecture",
        "chapter": "Chapter 3",
        "example": "Seq2seq models can translate English sentences to French by encoding English and decoding French."
    },
    "Softmax": {
        "definition": "A function that converts a vector of real numbers into a probability distribution, commonly used in the output layer of classifiers.",
        "category": "Mathematics",
        "chapter": "Chapter 2",
        "example": "Softmax ensures that predicted word probabilities in a language model sum to 1."
    },
    "Subword Tokenization": {
        "definition": "Breaking words into smaller units (subwords) to handle rare words and reduce vocabulary size.",
        "category": "Preprocessing",
        "chapter": "Chapter 4",
        "example": "The word 'unhappiness' might be tokenized as 'un', 'happy', 'ness'."
    },
    "TF-IDF": {
        "definition": "Term Frequency-Inverse Document Frequency. A weighting scheme that reflects how important a word is to a document in a collection.",
        "category": "Information Retrieval",
        "chapter": "Chapter 0",
        "example": "TF-IDF gives high weights to words that appear frequently in a document but rarely in the collection."
    },
    "Tokenization": {
        "definition": "The process of breaking text into individual units (tokens) such as words, subwords, or characters.",
        "category": "Preprocessing",
        "chapter": "Chapter 1",
        "example": "Tokenizing 'Hello, world!' might produce ['Hello', ',', 'world', '!']."
    },
    "Transfer Learning": {
        "definition": "Using knowledge gained from pre-training on one task to improve performance on a related task.",
        "category": "Machine Learning",
        "chapter": "Chapter 8",
        "example": "A model pre-trained on general text can be fine-tuned for medical text classification."
    },
    "Transformer": {
        "definition": "A neural network architecture based entirely on attention mechanisms, without recurrence or convolution.",
        "category": "Architecture",
        "chapter": "Chapter 4",
        "example": "The Transformer architecture enabled breakthrough models like BERT and GPT."
    },
    "Vanishing Gradient": {
        "definition": "A problem where gradients become exponentially small in deep networks, making it difficult to train early layers.",
        "category": "Training",
        "chapter": "Chapter 3",
        "example": "Standard RNNs suffer from vanishing gradients when processing long sequences."
    },
    "Vocabulary": {
        "definition": "The set of unique words or tokens that a model can understand and work with.",
        "category": "Preprocessing",
        "chapter": "Chapter 1",
        "example": "A model with a 50,000-word vocabulary can only process those specific tokens."
    },
    "Word2Vec": {
        "definition": "A method for learning word embeddings by predicting words from their context (CBOW) or context from words (Skip-gram).",
        "category": "Embeddings",
        "chapter": "Chapter 2",
        "example": "Word2Vec learns that 'king' - 'man' + 'woman' ≈ 'queen' in embedding space."
    },
    "Zero-Shot Learning": {
        "definition": "The ability of a model to perform tasks it wasn't explicitly trained on, using only task descriptions or examples.",
        "category": "Applications",
        "chapter": "Chapter 8",
        "example": "GPT-3 can perform translation without being explicitly trained on translation data."
    }
}

def render_glossary():
    """Render the comprehensive NLP glossary."""
    st.header("📚 NLP Glossary")
    
    st.markdown("""
    This comprehensive glossary covers key terms from the evolution of Natural Language Processing, 
    from early statistical methods to modern large language models.
    """)
    
    # Search functionality for the glossary
    search_term = st.text_input("🔍 Search terms:", placeholder="Type to search for a term...")
    
    # Category filter
    categories = sorted(set(term_data["category"] for term_data in NLP_TERMS.values()))
    selected_category = st.selectbox("Filter by category:", ["All"] + categories)
    
    # Chapter filter
    chapters = sorted(set(term_data["chapter"] for term_data in NLP_TERMS.values()))
    selected_chapter = st.selectbox("Filter by chapter:", ["All"] + chapters)
    
    # Filter terms based on search and filters
    filtered_terms = {}
    for term, data in NLP_TERMS.items():
        # Search filter
        if search_term and search_term.lower() not in term.lower() and search_term.lower() not in data["definition"].lower():
            continue
        
        # Category filter
        if selected_category != "All" and data["category"] != selected_category:
            continue
            
        # Chapter filter
        if selected_chapter != "All" and data["chapter"] != selected_chapter:
            continue
            
        filtered_terms[term] = data
    
    st.markdown(f"**Showing {len(filtered_terms)} of {len(NLP_TERMS)} terms**")
    
    # Display terms
    if filtered_terms:
        # Sort terms alphabetically
        sorted_terms = sorted(filtered_terms.items())
        
        # Create tabs for better organization
        if len(sorted_terms) > 20:
            # Group by first letter for very long lists
            letters = {}
            for term, data in sorted_terms:
                first_letter = term[0].upper()
                if first_letter not in letters:
                    letters[first_letter] = []
                letters[first_letter].append((term, data))
            
            # Create tabs for each letter that has terms
            if len(letters) > 1:
                tab_names = sorted(letters.keys())
                tabs = st.tabs(tab_names)
                
                for i, letter in enumerate(tab_names):
                    with tabs[i]:
                        for term, data in letters[letter]:
                            render_term_card(term, data)
            else:
                # Single tab if only one letter
                for term, data in sorted_terms:
                    render_term_card(term, data)
        else:
            # Display all terms if list is manageable
            for term, data in sorted_terms:
                render_term_card(term, data)
    else:
        st.info("No terms found matching your search criteria.")
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 Glossary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Terms", len(NLP_TERMS))
    
    with col2:
        category_counts = {}
        for data in NLP_TERMS.values():
            category = data["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        most_common_category = max(category_counts, key=category_counts.get)
        st.metric("Categories", len(categories))
        st.caption(f"Most terms: {most_common_category}")
    
    with col3:
        chapter_counts = {}
        for data in NLP_TERMS.values():
            chapter = data["chapter"]
            chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
        st.metric("Chapters Covered", len(chapters))
    
    # Category breakdown
    with st.expander("📈 Terms by Category"):
        category_df = pd.DataFrame([
            {"Category": cat, "Count": count} 
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(category_df, hide_index=True)

def render_term_card(term: str, data: dict):
    """Render a single term card."""
    with st.expander(f"**{term}**"):
        st.markdown(f"**Definition:** {data['definition']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Category:** {data['category']}")
        with col2:
            st.markdown(f"**Chapter:** {data['chapter']}")
        
        if data.get("example"):
            st.markdown(f"**Example:** {data['example']}")

if __name__ == "__main__":
    render_glossary()