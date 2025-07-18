"""
Interactive code completion and exercises for NLP Evolution app.
Provides hands-on coding practice with fill-in-the-blank exercises.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
import random
from utils import handle_errors

# Code exercise templates organized by chapter
CODE_EXERCISES = {
    "chapter1": {
        "title": "N-gram Models",
        "exercises": [
            {
                "id": "ngram_basic",
                "title": "Build a Simple Bigram Model",
                "description": "Complete the code to build a bigram model from text.",
                "template": '''
import re
from collections import defaultdict, Counter

def build_bigram_model(text):
    """Build a bigram model from text."""
    # Tokenize and clean the text
    words = re.findall(r'\\w+', text.lower())
    
    # Add start and end tokens
    words = ['<start>'] + words + ['<end>']
    
    # Initialize bigram counts
    bigram_counts = defaultdict(Counter)
    
    # Count bigrams
    for i in range(len(words) - 1):
        first_word = ___BLANK1___
        second_word = ___BLANK2___
        bigram_counts[first_word][second_word] += 1
    
    return bigram_counts

def get_next_word_probabilities(model, word):
    """Get probability distribution for next word."""
    if word not in model:
        return {}
    
    total_count = ___BLANK3___
    probabilities = {}
    
    for next_word, count in model[word].items():
        probabilities[next_word] = ___BLANK4___
    
    return probabilities

# Test the model
text = "the cat sat on the mat the dog ran in the park"
model = build_bigram_model(text)
probs = get_next_word_probabilities(model, "the")
print(probs)
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "words[i]",
                        "options": ["words[i]", "words[i+1]", "words[i-1]", "i"],
                        "explanation": "The first word in the bigram is at position i."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "words[i+1]",
                        "options": ["words[i]", "words[i+1]", "words[i-1]", "i+1"],
                        "explanation": "The second word in the bigram is at position i+1."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "sum(model[word].values())",
                        "options": ["len(model[word])", "sum(model[word].values())", "model[word].total()", "count"],
                        "explanation": "Total count is the sum of all counts for this word."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "count / total_count",
                        "options": ["count / total_count", "count", "total_count / count", "count + total_count"],
                        "explanation": "Probability is count divided by total count."
                    }
                ]
            },
            {
                "id": "smoothing",
                "title": "Add-1 Smoothing",
                "description": "Implement add-1 (Laplace) smoothing for n-gram models.",
                "template": '''
def add_one_smoothing(bigram_counts, vocabulary_size):
    """Apply add-1 smoothing to bigram counts."""
    smoothed_probs = {}
    
    for first_word in bigram_counts:
        smoothed_probs[first_word] = {}
        
        # Calculate total count for this first word
        total_count = ___BLANK1___
        
        for second_word in bigram_counts[first_word]:
            original_count = bigram_counts[first_word][second_word]
            
            # Apply add-1 smoothing
            smoothed_count = ___BLANK2___
            smoothed_total = ___BLANK3___
            
            smoothed_probs[first_word][second_word] = ___BLANK4___
    
    return smoothed_probs

# Test smoothing
vocabulary_size = 1000
smoothed = add_one_smoothing(bigram_counts, vocabulary_size)
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "sum(bigram_counts[first_word].values())",
                        "options": ["len(bigram_counts[first_word])", "sum(bigram_counts[first_word].values())", "vocabulary_size", "bigram_counts[first_word]"],
                        "explanation": "Total count is the sum of all bigram counts for this first word."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "original_count + 1",
                        "options": ["original_count + 1", "original_count", "original_count * 1", "1"],
                        "explanation": "Add-1 smoothing adds 1 to the original count."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "total_count + vocabulary_size",
                        "options": ["total_count + vocabulary_size", "total_count + 1", "vocabulary_size", "total_count"],
                        "explanation": "The denominator includes the vocabulary size in add-1 smoothing."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "smoothed_count / smoothed_total",
                        "options": ["smoothed_count / smoothed_total", "smoothed_count", "original_count / total_count", "smoothed_total"],
                        "explanation": "Smoothed probability is smoothed count divided by smoothed total."
                    }
                ]
            }
        ]
    },
    
    "chapter2": {
        "title": "Word Embeddings",
        "exercises": [
            {
                "id": "cosine_similarity",
                "title": "Calculate Cosine Similarity",
                "description": "Implement cosine similarity to find similar words in embedding space.",
                "template": '''
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    # Calculate dot product
    dot_product = ___BLANK1___
    
    # Calculate norms
    norm1 = ___BLANK2___
    norm2 = ___BLANK3___
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0
    
    # Calculate cosine similarity
    similarity = ___BLANK4___
    
    return similarity

def find_most_similar(target_word, word_vectors, top_k=5):
    """Find most similar words to target word."""
    if target_word not in word_vectors:
        return []
    
    target_vector = word_vectors[target_word]
    similarities = []
    
    for word, vector in word_vectors.items():
        if word != target_word:
            sim = cosine_similarity(target_vector, vector)
            similarities.append((word, sim))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: ___BLANK5___, reverse=True)
    return similarities[:top_k]

# Test with sample vectors
word_vectors = {
    'king': np.array([0.1, 0.3, -0.5]),
    'queen': np.array([0.2, 0.2, -0.4]),
    'man': np.array([0.1, 0.5, -0.6]),
    'woman': np.array([0.3, 0.4, -0.3])
}

similar = find_most_similar('king', word_vectors)
print(similar)
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "np.dot(vec1, vec2)",
                        "options": ["np.dot(vec1, vec2)", "vec1 * vec2", "vec1 + vec2", "np.sum(vec1, vec2)"],
                        "explanation": "Dot product is calculated using np.dot()."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "np.linalg.norm(vec1)",
                        "options": ["np.linalg.norm(vec1)", "np.sum(vec1)", "np.sqrt(vec1)", "len(vec1)"],
                        "explanation": "Vector norm is calculated using np.linalg.norm()."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "np.linalg.norm(vec2)",
                        "options": ["np.linalg.norm(vec2)", "np.sum(vec2)", "np.sqrt(vec2)", "len(vec2)"],
                        "explanation": "Vector norm is calculated using np.linalg.norm()."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "dot_product / (norm1 * norm2)",
                        "options": ["dot_product / (norm1 * norm2)", "dot_product / norm1", "dot_product * norm1 * norm2", "(norm1 * norm2) / dot_product"],
                        "explanation": "Cosine similarity is dot product divided by the product of norms."
                    },
                    {
                        "blank": "___BLANK5___",
                        "correct": "x[1]",
                        "options": ["x[1]", "x[0]", "x", "len(x)"],
                        "explanation": "Sort by the similarity score, which is the second element (index 1) of each tuple."
                    }
                ]
            },
            {
                "id": "word_analogy",
                "title": "Word Analogies with Vectors",
                "description": "Solve word analogies using vector arithmetic (king - man + woman = queen).",
                "template": '''
def solve_analogy(word_vectors, word_a, word_b, word_c, top_k=5):
    """Solve analogy: word_a is to word_b as word_c is to ?"""
    
    # Check if all words are in vocabulary
    required_words = [word_a, word_b, word_c]
    for word in required_words:
        if word not in word_vectors:
            return f"Word '{word}' not in vocabulary"
    
    # Get vectors
    vec_a = ___BLANK1___
    vec_b = ___BLANK2___
    vec_c = ___BLANK3___
    
    # Calculate target vector: b - a + c
    target_vector = ___BLANK4___
    
    # Find most similar words to target vector
    similarities = []
    exclude_words = {word_a, word_b, word_c}
    
    for word, vector in word_vectors.items():
        if word not in exclude_words:
            similarity = cosine_similarity(target_vector, vector)
            similarities.append((word, similarity))
    
    # Sort and return top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def vector_arithmetic_demo():
    """Demonstrate vector arithmetic properties."""
    # Example: king - man + woman ≈ queen
    result = solve_analogy(word_vectors, 'king', 'man', 'woman')
    
    print("king - man + woman =", result)
    
    # The analogy can be read as:
    # king is to man as ? is to woman
    # Answer should be close to queen

# Test the analogy
result = solve_analogy(word_vectors, 'king', 'man', 'woman')
print(result)
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "word_vectors[word_a]",
                        "options": ["word_vectors[word_a]", "word_a", "word_vectors.get(word_a)", "word_vectors[0]"],
                        "explanation": "Get the vector for word_a from the word_vectors dictionary."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "word_vectors[word_b]",
                        "options": ["word_vectors[word_b]", "word_b", "word_vectors.get(word_b)", "word_vectors[1]"],
                        "explanation": "Get the vector for word_b from the word_vectors dictionary."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "word_vectors[word_c]",
                        "options": ["word_vectors[word_c]", "word_c", "word_vectors.get(word_c)", "word_vectors[2]"],
                        "explanation": "Get the vector for word_c from the word_vectors dictionary."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "vec_b - vec_a + vec_c",
                        "options": ["vec_b - vec_a + vec_c", "vec_a - vec_b + vec_c", "vec_a + vec_b - vec_c", "vec_c - vec_b + vec_a"],
                        "explanation": "For analogy 'a is to b as c is to ?', the formula is b - a + c."
                    }
                ]
            }
        ]
    },
    
    "chapter3": {
        "title": "RNNs and LSTMs",
        "exercises": [
            {
                "id": "simple_rnn",
                "title": "Simple RNN Forward Pass",
                "description": "Implement a basic RNN forward pass.",
                "template": '''
import numpy as np

def rnn_forward_step(x_t, h_prev, W_xh, W_hh, b_h):
    """Single forward step of RNN."""
    
    # Calculate new hidden state
    # h_t = tanh(W_xh * x_t + W_hh * h_prev + b_h)
    
    linear_output = ___BLANK1___ + ___BLANK2___ + ___BLANK3___
    h_t = ___BLANK4___
    
    return h_t

def rnn_forward_sequence(X, h_0, W_xh, W_hh, b_h):
    """Forward pass through entire sequence."""
    
    sequence_length, input_size = X.shape
    hidden_size = h_0.shape[0]
    
    # Store all hidden states
    hidden_states = np.zeros((sequence_length, hidden_size))
    h_t = h_0
    
    for t in range(sequence_length):
        x_t = ___BLANK5___
        h_t = rnn_forward_step(x_t, h_t, W_xh, W_hh, b_h)
        hidden_states[t] = h_t
    
    return hidden_states

# Test RNN
sequence_length, input_size, hidden_size = 5, 3, 4

# Initialize parameters
W_xh = np.random.randn(hidden_size, input_size) * 0.1
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
b_h = np.zeros((hidden_size, 1))
h_0 = np.zeros((hidden_size, 1))

# Sample input sequence
X = np.random.randn(sequence_length, input_size)

# Forward pass
hidden_states = rnn_forward_sequence(X, h_0, W_xh, W_hh, b_h)
print("Hidden states shape:", hidden_states.shape)
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "W_xh @ x_t",
                        "options": ["W_xh @ x_t", "x_t @ W_xh", "W_xh * x_t", "np.dot(x_t, W_xh)"],
                        "explanation": "Matrix multiplication of weight matrix W_xh with input x_t."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "W_hh @ h_prev",
                        "options": ["W_hh @ h_prev", "h_prev @ W_hh", "W_hh * h_prev", "np.dot(h_prev, W_hh)"],
                        "explanation": "Matrix multiplication of recurrent weight matrix W_hh with previous hidden state."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "b_h",
                        "options": ["b_h", "b_h.T", "np.transpose(b_h)", "b_h @ 1"],
                        "explanation": "Add the bias term b_h."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "np.tanh(linear_output)",
                        "options": ["np.tanh(linear_output)", "np.sigmoid(linear_output)", "np.relu(linear_output)", "linear_output"],
                        "explanation": "Apply tanh activation function to get the hidden state."
                    },
                    {
                        "blank": "___BLANK5___",
                        "correct": "X[t].reshape(-1, 1)",
                        "options": ["X[t].reshape(-1, 1)", "X[t]", "X[:, t]", "X[t, :]"],
                        "explanation": "Get the input at time step t and reshape it to column vector."
                    }
                ]
            }
        ]
    },
    
    "chapter4": {
        "title": "Attention and Transformers",
        "exercises": [
            {
                "id": "attention_basic",
                "title": "Implement Scaled Dot-Product Attention",
                "description": "Complete the implementation of the attention mechanism.",
                "template": '''
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Optional attention mask
    
    Returns:
        output: Attention output
        attention_weights: Attention weights
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = ___BLANK1___
    
    # Scale scores
    scores = scores / ___BLANK2___
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, ___BLANK3___)
    
    # Apply softmax
    attention_weights = ___BLANK4___
    
    # Apply attention to values
    output = ___BLANK5___
    
    return output, attention_weights

# Test the implementation
seq_len, d_k, d_v = 4, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "np.matmul(Q, K.T)",
                        "options": ["np.matmul(Q, K.T)", "np.dot(Q, K)", "Q @ K", "Q * K"],
                        "explanation": "Attention scores are computed as the dot product of queries and keys (transposed)."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "np.sqrt(d_k)",
                        "options": ["np.sqrt(d_k)", "d_k", "d_k ** 2", "np.log(d_k)"],
                        "explanation": "Scores are scaled by the square root of the key dimension to prevent saturation."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "-1e9",
                        "options": ["-1e9", "0", "1", "float('inf')"],
                        "explanation": "Masked positions are set to a large negative value so they become ~0 after softmax."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)",
                        "options": [
                            "np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)",
                            "np.softmax(scores)",
                            "scores / np.sum(scores)",
                            "np.sigmoid(scores)"
                        ],
                        "explanation": "Softmax normalizes scores to get attention weights that sum to 1."
                    },
                    {
                        "blank": "___BLANK5___",
                        "correct": "np.matmul(attention_weights, V)",
                        "options": ["np.matmul(attention_weights, V)", "attention_weights * V", "V @ attention_weights", "np.dot(V, attention_weights)"],
                        "explanation": "The output is computed by applying attention weights to the values."
                    }
                ]
            },
            {
                "id": "positional_encoding",
                "title": "Implement Positional Encoding",
                "description": "Add positional information to embeddings.",
                "template": '''
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Create positional encoding for transformer.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
    
    Returns:
        PE: Positional encoding matrix (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))
    
    # Create position array
    position = ___BLANK1___
    
    # Create dimension array
    div_term = np.exp(np.arange(0, d_model, 2) * ___BLANK2___)
    
    # Apply sin to even indices
    PE[:, 0::2] = ___BLANK3___
    
    # Apply cos to odd indices
    PE[:, 1::2] = ___BLANK4___
    
    return PE

# Test positional encoding
seq_len, d_model = 10, 16
pe = positional_encoding(seq_len, d_model)
print(f"Positional encoding shape: {pe.shape}")

# Visualize
import matplotlib.pyplot as plt
plt.imshow(pe, aspect='auto', cmap='RdBu')
plt.colorbar()
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding')
plt.show()
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "np.arange(seq_len).reshape(-1, 1)",
                        "options": ["np.arange(seq_len).reshape(-1, 1)", "np.arange(seq_len)", "np.ones(seq_len)", "np.linspace(0, seq_len, seq_len)"],
                        "explanation": "Position array contains indices from 0 to seq_len-1 as a column vector."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "-(np.log(10000.0) / d_model)",
                        "options": ["-(np.log(10000.0) / d_model)", "-np.log(10000.0)", "1.0 / d_model", "np.log(d_model)"],
                        "explanation": "This creates the frequency scaling term for different dimensions."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "np.sin(position * div_term)",
                        "options": ["np.sin(position * div_term)", "np.sin(position + div_term)", "np.sin(position)", "position * np.sin(div_term)"],
                        "explanation": "Sine is applied to even dimensions with position-dependent frequency."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "np.cos(position * div_term)",
                        "options": ["np.cos(position * div_term)", "np.cos(position + div_term)", "np.cos(position)", "position * np.cos(div_term)"],
                        "explanation": "Cosine is applied to odd dimensions with position-dependent frequency."
                    }
                ]
            }
        ]
    },
    
    "chapter5": {
        "title": "Text Classification",
        "exercises": [
            {
                "id": "tfidf_classifier",
                "title": "Build a TF-IDF Text Classifier",
                "description": "Implement a simple text classifier using TF-IDF features.",
                "template": '''
# Use fallback implementations to avoid Python 3.13 sklearn compatibility issues
from sklearn_fallbacks import TfidfVectorizer, MultinomialNB, train_test_split, accuracy_score

# Sample data
texts = [
    "I love this movie", "Great film", "Terrible movie",
    "Worst film ever", "Amazing cinematography", "Boring plot"
]
labels = ["positive", "positive", "negative", "negative", "positive", "negative"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(___BLANK1___)

# Transform training data
X_train_tfidf = ___BLANK2___

# Transform test data
X_test_tfidf = ___BLANK3___

# Create and train classifier
classifier = ___BLANK4___
classifier.fit(X_train_tfidf, y_train)

# Make predictions
predictions = ___BLANK5___

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "max_features=100",
                        "options": ["max_features=100", "max_features='auto'", "min_df=2", "ngram_range=(1,1)"],
                        "explanation": "max_features limits the vocabulary size to the most frequent terms."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "vectorizer.fit_transform(X_train)",
                        "options": ["vectorizer.fit_transform(X_train)", "vectorizer.transform(X_train)", "vectorizer.fit(X_train)", "TfidfVectorizer(X_train)"],
                        "explanation": "fit_transform learns the vocabulary and transforms training data in one step."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "vectorizer.transform(X_test)",
                        "options": ["vectorizer.transform(X_test)", "vectorizer.fit_transform(X_test)", "vectorizer.fit(X_test)", "X_test"],
                        "explanation": "Use transform (not fit_transform) on test data to use the same vocabulary as training."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "MultinomialNB()",
                        "options": ["MultinomialNB()", "GaussianNB()", "BernoulliNB()", "ComplementNB()"],
                        "explanation": "MultinomialNB works well with TF-IDF features for text classification."
                    },
                    {
                        "blank": "___BLANK5___",
                        "correct": "classifier.predict(X_test_tfidf)",
                        "options": ["classifier.predict(X_test_tfidf)", "classifier.predict_proba(X_test_tfidf)", "classifier.transform(X_test_tfidf)", "classifier.fit_predict(X_test_tfidf)"],
                        "explanation": "predict() returns the predicted class labels for the test data."
                    }
                ]
            }
        ]
    },
    
    "chapter6": {
        "title": "Text Generation",
        "exercises": [
            {
                "id": "beam_search",
                "title": "Implement Beam Search Decoding",
                "description": "Complete the beam search algorithm for text generation.",
                "template": '''
import numpy as np

def beam_search(model, start_token, end_token, beam_width=3, max_length=20):
    """
    Implement beam search decoding.
    
    Args:
        model: Language model with predict_next method
        start_token: Starting token ID
        end_token: End token ID
        beam_width: Number of beams to maintain
        max_length: Maximum sequence length
    
    Returns:
        best_sequence: Best sequence found
    """
    # Initialize beams with start token
    beams = [[start_token]]
    beam_scores = [0.0]
    
    for step in range(max_length):
        all_candidates = []
        
        # Expand each beam
        for i, beam in enumerate(beams):
            # Skip if beam ended
            if beam[-1] == end_token:
                all_candidates.append((beam_scores[i], beam))
                continue
            
            # Get next token probabilities
            next_probs = model.predict_next(beam)
            
            # Get top k tokens
            top_k_tokens = ___BLANK1___
            top_k_probs = ___BLANK2___
            
            # Create candidates
            for token, prob in zip(top_k_tokens, top_k_probs):
                candidate = beam + [token]
                # Score is sum of log probabilities
                score = beam_scores[i] + ___BLANK3___
                all_candidates.append((score, candidate))
        
        # Select top beam_width candidates
        all_candidates.sort(key=lambda x: ___BLANK4___, reverse=True)
        
        # Update beams
        beams = []
        beam_scores = []
        for score, candidate in all_candidates[:beam_width]:
            beams.append(candidate)
            beam_scores.append(score)
        
        # Check if all beams have ended
        if all(beam[-1] == end_token for beam in beams):
            break
    
    # Return best beam
    best_idx = ___BLANK5___
    return beams[best_idx]

# Example usage (pseudo-code)
# model = YourLanguageModel()
# sequence = beam_search(model, start_token=0, end_token=1)
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "np.argsort(next_probs)[-beam_width:]",
                        "options": ["np.argsort(next_probs)[-beam_width:]", "np.argmax(next_probs)", "next_probs[:beam_width]", "np.random.choice(len(next_probs), beam_width)"],
                        "explanation": "Get indices of top k tokens by probability using argsort."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "next_probs[top_k_tokens]",
                        "options": ["next_probs[top_k_tokens]", "next_probs[:beam_width]", "top_k_tokens", "np.max(next_probs)"],
                        "explanation": "Get the actual probability values for the top k tokens."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "np.log(prob)",
                        "options": ["np.log(prob)", "prob", "np.exp(prob)", "-np.log(prob)"],
                        "explanation": "Use log probabilities to avoid underflow and make scores additive."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "x[0]",
                        "options": ["x[0]", "x[1]", "len(x[1])", "-x[0]"],
                        "explanation": "Sort by score (first element of tuple) in descending order."
                    },
                    {
                        "blank": "___BLANK5___",
                        "correct": "np.argmax(beam_scores)",
                        "options": ["np.argmax(beam_scores)", "0", "-1", "np.argmin(beam_scores)"],
                        "explanation": "Select the beam with the highest score."
                    }
                ]
            }
        ]
    },
    
    "chapter7": {
        "title": "Building Your Own Model",
        "exercises": [
            {
                "id": "transformer_block",
                "title": "Build a Transformer Block",
                "description": "Implement a single transformer encoder block.",
                "template": '''
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Single transformer encoder block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            ___BLANK1___,
            ___BLANK2___,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ___BLANK3___),
            nn.ReLU(),
            nn.Linear(___BLANK4___, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = ___BLANK5___
        x = self.dropout(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Test the transformer block
d_model, n_heads, d_ff = 512, 8, 2048
seq_len, batch_size = 10, 32

block = TransformerBlock(d_model, n_heads, d_ff)
x = torch.randn(seq_len, batch_size, d_model)
output = block(x)
print(f"Output shape: {output.shape}")
''',
                "blanks": [
                    {
                        "blank": "___BLANK1___",
                        "correct": "d_model",
                        "options": ["d_model", "n_heads", "d_ff", "d_model * n_heads"],
                        "explanation": "MultiheadAttention takes the model dimension as the first argument."
                    },
                    {
                        "blank": "___BLANK2___",
                        "correct": "n_heads",
                        "options": ["n_heads", "d_model", "d_model // n_heads", "1"],
                        "explanation": "Number of attention heads is the second argument."
                    },
                    {
                        "blank": "___BLANK3___",
                        "correct": "d_ff",
                        "options": ["d_ff", "d_model", "d_model * 4", "n_heads"],
                        "explanation": "First linear layer expands to feed-forward dimension."
                    },
                    {
                        "blank": "___BLANK4___",
                        "correct": "d_ff",
                        "options": ["d_ff", "d_model", "n_heads", "d_model // 2"],
                        "explanation": "Second linear layer takes d_ff as input dimension."
                    },
                    {
                        "blank": "___BLANK5___",
                        "correct": "self.norm1(x + attn_output)",
                        "options": ["self.norm1(x + attn_output)", "self.norm1(attn_output)", "x + self.norm1(attn_output)", "attn_output"],
                        "explanation": "Apply layer norm to the sum of input and attention output (residual connection)."
                    }
                ]
            }
        ]
    }
}

class CodeExerciseManager:
    """Manages interactive code completion exercises."""
    
    def __init__(self):
        self.exercises = CODE_EXERCISES
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for exercise tracking."""
        if "exercise_progress" not in st.session_state:
            st.session_state.exercise_progress = {}
        if "current_exercise" not in st.session_state:
            st.session_state.current_exercise = None
    
    @handle_errors
    def get_exercises_for_chapter(self, chapter: str) -> Optional[Dict]:
        """Get exercises for a specific chapter."""
        return self.exercises.get(chapter, None)
    
    @handle_errors
    def start_exercise(self, chapter: str, exercise_id: str) -> bool:
        """Start a specific exercise."""
        if chapter not in self.exercises:
            return False
        
        chapter_exercises = self.exercises[chapter]["exercises"]
        exercise = next((ex for ex in chapter_exercises if ex["id"] == exercise_id), None)
        
        if not exercise:
            return False
        
        st.session_state.current_exercise = {
            "chapter": chapter,
            "exercise": exercise,
            "user_answers": {},
            "completed": False
        }
        
        return True
    
    @handle_errors
    def submit_answer(self, blank_id: str, answer: str) -> bool:
        """Submit an answer for a blank."""
        if not st.session_state.current_exercise:
            return False
        
        st.session_state.current_exercise["user_answers"][blank_id] = answer
        return True
    
    @handle_errors
    def check_answers(self) -> Dict:
        """Check all answers and return results."""
        if not st.session_state.current_exercise:
            return {}
        
        exercise = st.session_state.current_exercise["exercise"]
        user_answers = st.session_state.current_exercise["user_answers"]
        
        results = {}
        correct_count = 0
        
        for blank in exercise["blanks"]:
            blank_id = blank["blank"]
            correct_answer = blank["correct"]
            user_answer = user_answers.get(blank_id, "")
            
            is_correct = user_answer.strip() == correct_answer.strip()
            if is_correct:
                correct_count += 1
            
            results[blank_id] = {
                "is_correct": is_correct,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "explanation": blank["explanation"]
            }
        
        total_blanks = len(exercise["blanks"])
        score = (correct_count / total_blanks) * 100
        
        # Save progress
        chapter = st.session_state.current_exercise["chapter"]
        exercise_id = exercise["id"]
        
        if chapter not in st.session_state.exercise_progress:
            st.session_state.exercise_progress[chapter] = {}
        
        st.session_state.exercise_progress[chapter][exercise_id] = {
            "score": score,
            "correct": correct_count,
            "total": total_blanks,
            "completed": True
        }
        
        st.session_state.current_exercise["completed"] = True
        
        return {
            "score": score,
            "correct": correct_count,
            "total": total_blanks,
            "details": results
        }
    
    @handle_errors
    def get_completed_code(self) -> str:
        """Get the code with user answers filled in."""
        if not st.session_state.current_exercise:
            return ""
        
        exercise = st.session_state.current_exercise["exercise"]
        user_answers = st.session_state.current_exercise["user_answers"]
        
        code = exercise["template"]
        
        for blank in exercise["blanks"]:
            blank_id = blank["blank"]
            user_answer = user_answers.get(blank_id, blank_id)
            code = code.replace(blank_id, user_answer)
        
        return code

def render_code_exercise_interface():
    """Render the main code exercise interface."""
    st.subheader("💻 Interactive Code Exercises")
    
    exercise_manager = CodeExerciseManager()
    
    # Check if there's an active exercise
    if st.session_state.current_exercise and not st.session_state.current_exercise["completed"]:
        render_active_exercise(exercise_manager)
    else:
        render_exercise_selection(exercise_manager)

def render_exercise_selection(exercise_manager: CodeExerciseManager):
    """Render exercise selection interface."""
    st.markdown("Practice your coding skills with interactive fill-in-the-blank exercises!")
    
    # Available exercises by chapter
    for chapter, chapter_data in exercise_manager.exercises.items():
        st.markdown(f"### {chapter_data['title']}")
        
        for exercise in chapter_data["exercises"]:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{exercise['title']}**")
                st.write(exercise["description"])
            
            with col2:
                # Show progress if available
                progress = st.session_state.exercise_progress.get(chapter, {}).get(exercise["id"], None)
                if progress:
                    st.success(f"✅ {progress['score']:.0f}%")
                else:
                    st.info("Not started")
            
            with col3:
                if st.button("Start", key=f"start_{chapter}_{exercise['id']}"):
                    if exercise_manager.start_exercise(chapter, exercise["id"]):
                        st.rerun()

def render_active_exercise(exercise_manager: CodeExerciseManager):
    """Render active exercise interface."""
    exercise_data = st.session_state.current_exercise["exercise"]
    
    st.markdown(f"## {exercise_data['title']}")
    st.markdown(exercise_data["description"])
    
    # Show the code template with blanks
    st.markdown("### Complete the Code")
    st.markdown("Fill in the blanks to complete the implementation:")
    
    # Display code with input fields for blanks
    code_lines = exercise_data["template"].split('\n')
    
    for blank in exercise_data["blanks"]:
        blank_id = blank["blank"]
        
        st.markdown(f"**{blank_id}:**")
        
        # Multiple choice for easier completion
        if "options" in blank:
            selected_option = st.radio(
                f"Choose the correct completion for {blank_id}:",
                blank["options"],
                key=f"blank_{blank_id}",
                horizontal=True
            )
            exercise_manager.submit_answer(blank_id, selected_option)
        else:
            # Text input for free-form answers
            user_input = st.text_input(
                f"Fill in {blank_id}:",
                key=f"blank_{blank_id}"
            )
            exercise_manager.submit_answer(blank_id, user_input)
    
    # Show current code with answers
    st.markdown("### Your Code")
    completed_code = exercise_manager.get_completed_code()
    st.code(completed_code, language="python")
    
    # Submit button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Check Answers", type="primary"):
            results = exercise_manager.check_answers()
            render_exercise_results(results)
    
    with col2:
        if st.button("Exit Exercise"):
            st.session_state.current_exercise = None
            st.rerun()

def render_exercise_results(results: Dict):
    """Render exercise results."""
    st.markdown("## 📊 Exercise Results")
    
    # Overall score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score", f"{results['score']:.0f}%")
    with col2:
        st.metric("Correct", f"{results['correct']}/{results['total']}")
    with col3:
        if results['score'] >= 80:
            st.success("🌟 Excellent!")
        elif results['score'] >= 60:
            st.info("👍 Good job!")
        else:
            st.warning("📚 Keep practicing!")
    
    # Detailed feedback
    st.markdown("### Detailed Feedback")
    
    for blank_id, result in results["details"].items():
        with st.expander(f"{blank_id}: {'✅' if result['is_correct'] else '❌'}"):
            st.write(f"**Your answer:** `{result['user_answer']}`")
            
            if not result["is_correct"]:
                st.write(f"**Correct answer:** `{result['correct_answer']}`")
            
            st.write(f"**Explanation:** {result['explanation']}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Try Again"):
            # Reset the exercise
            exercise = st.session_state.current_exercise["exercise"]
            chapter = st.session_state.current_exercise["chapter"]
            exercise_manager = CodeExerciseManager()
            exercise_manager.start_exercise(chapter, exercise["id"])
            st.rerun()
    
    with col2:
        if st.button("Back to Exercises"):
            st.session_state.current_exercise = None
            st.rerun()

def render_exercise_widget(chapter: str):
    """Render a compact exercise widget for a specific chapter."""
    exercise_manager = CodeExerciseManager()
    
    # Check if there's an active exercise from this chapter
    if (st.session_state.current_exercise and 
        st.session_state.current_exercise.get("chapter") == chapter and 
        not st.session_state.current_exercise.get("completed", False)):
        
        # Render the active exercise
        render_active_exercise(exercise_manager)
        return
    
    # Check if exercises exist for this chapter
    chapter_exercises = exercise_manager.get_exercises_for_chapter(chapter)
    if not chapter_exercises:
        return
    
    st.markdown("### 💻 Coding Practice")
    
    exercises = chapter_exercises["exercises"]
    
    # Show available exercises
    for exercise in exercises:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{exercise['title']}**")
            st.caption(exercise['description'])
        
        with col2:
            if st.button("Start", key=f"widget_{chapter}_{exercise['id']}"):
                if exercise_manager.start_exercise(chapter, exercise["id"]):
                    st.rerun()

def get_exercise_progress(chapter: str) -> Dict:
    """Get exercise progress for a specific chapter."""
    if "exercise_progress" not in st.session_state:
        return {}
    
    return st.session_state.exercise_progress.get(chapter, {})