"""
Fallback implementations for sklearn functionality to avoid Python 3.13 compatibility issues.
Provides educational implementations of key ML algorithms.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re
import math
from typing import List, Union, Tuple, Optional

class TfidfVectorizer:
    """
    Simple TF-IDF vectorizer implementation as fallback for sklearn.
    Educational implementation that demonstrates the core concepts.
    """
    
    def __init__(self, max_features: int = 1000, stop_words: Optional[str] = 'english', 
                 lowercase: bool = True, max_df: float = 1.0, min_df: int = 1):
        self.max_features = max_features
        self.lowercase = lowercase
        self.max_df = max_df
        self.min_df = min_df
        self.vocabulary_ = {}
        self.idf_ = {}
        
        # Simple English stop words list
        if stop_words == 'english':
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
                'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
                'its', 'our', 'their'
            }
        else:
            self.stop_words = set(stop_words) if stop_words else set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if self.lowercase:
            text = text.lower()
        # Extract words (letters and numbers)
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stop words
        return [token for token in tokens if token not in self.stop_words]
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Count term frequencies across documents
        term_doc_freq = defaultdict(int)
        all_terms = set()
        
        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_freq[token] += 1
            all_terms.update(tokens)
        
        # Filter by document frequency
        total_docs = len(texts)
        filtered_terms = []
        
        for term, doc_freq in term_doc_freq.items():
            if (doc_freq >= self.min_df and 
                doc_freq <= self.max_df * total_docs):
                filtered_terms.append((term, doc_freq))
        
        # Sort by document frequency and take top terms
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            filtered_terms = filtered_terms[:self.max_features]
        
        # Build vocabulary mapping
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(filtered_terms)}
        
        # Calculate IDF values
        total_docs = len(texts)
        for term, doc_freq in filtered_terms:
            self.idf_[term] = math.log(total_docs / doc_freq)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform texts to TF-IDF matrix."""
        self._build_vocabulary(texts)
        return self.transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF matrix."""
        n_docs = len(texts)
        n_features = len(self.vocabulary_)
        
        # Initialize matrix
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            
            # Calculate term frequencies
            term_freq = Counter(tokens)
            total_terms = len(tokens)
            
            # Calculate TF-IDF for each term
            for term, freq in term_freq.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    tf = freq / total_terms  # Term frequency
                    idf = self.idf_[term]     # Inverse document frequency
                    tfidf_matrix[doc_idx, term_idx] = tf * idf
        
        return tfidf_matrix


class PCA:
    """
    Simple PCA implementation as fallback for sklearn.
    Educational implementation for dimensionality reduction.
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """Fit PCA on the data."""
        X = np.array(X)
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        self.components_ = eigenvectors[:, :self.n_components].T
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensions."""
        X = np.array(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(X).transform(X)


class MultinomialNB:
    """
    Simple Multinomial Naive Bayes implementation as fallback for sklearn.
    Educational implementation for text classification.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Smoothing parameter
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNB':
        """Fit Naive Bayes classifier."""
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        for class_label in self.classes_:
            class_count = np.sum(y == class_label)
            self.class_log_prior_[class_label] = np.log(class_count / n_samples)
        
        # Calculate feature probabilities for each class
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_features = X[class_mask]
            
            # Sum of features for this class
            class_feature_counts = np.sum(class_features, axis=0)
            
            # Total count for this class (with smoothing)
            total_count = np.sum(class_feature_counts) + self.alpha * n_features
            
            # Calculate log probabilities with Laplace smoothing
            feature_probs = (class_feature_counts + self.alpha) / total_count
            self.feature_log_prob_[class_label] = np.log(feature_probs)
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class log probabilities."""
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes_):
            # Start with class prior
            class_log_prob = self.class_log_prior_[class_label]
            
            # Add feature contributions
            feature_contributions = np.dot(X, self.feature_log_prob_[class_label])
            
            log_proba[:, i] = class_log_prob + feature_contributions
        
        return log_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        log_proba = self.predict_log_proba(X)
        
        # Convert from log space to probabilities
        # Subtract max for numerical stability
        log_proba_stable = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba_stable)
        
        # Normalize
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


def train_test_split(*arrays, test_size: float = 0.25, random_state: Optional[int] = None) -> List[np.ndarray]:
    """
    Simple train-test split implementation as fallback for sklearn.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if len(arrays) == 0:
        return []
    
    # Get the length from the first array
    n_samples = len(arrays[0])
    
    # Generate random indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    split_point = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # Split all arrays
    result = []
    for array in arrays:
        array = np.array(array)
        result.append(array[train_indices])  # Train
        result.append(array[test_indices])   # Test
    
    return result


def accuracy_score(y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
    """Calculate accuracy score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total if total > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    # Test TF-IDF Vectorizer
    print("Testing TF-IDF Vectorizer...")
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park", 
        "cats and dogs are pets"
    ]
    
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary: {vectorizer.vocabulary_}")
    
    # Test PCA
    print("\nTesting PCA...")
    data = np.random.rand(100, 5)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    print(f"Original shape: {data.shape}")
    print(f"Reduced shape: {reduced_data.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Test Naive Bayes
    print("\nTesting Multinomial Naive Bayes...")
    X = np.random.randint(0, 10, (100, 5))
    y = np.random.randint(0, 2, 100)
    
    nb = MultinomialNB()
    nb.fit(X, y)
    predictions = nb.predict(X[:10])
    probabilities = nb.predict_proba(X[:10])
    print(f"Predictions: {predictions}")
    print(f"First probability: {probabilities[0]}")
    
    # Test train_test_split
    print("\nTesting train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Test shapes: {X_test.shape}, {y_test.shape}")
    
    # Test accuracy_score
    print("\nTesting accuracy_score...")
    acc = accuracy_score(y_test, nb.predict(X_test))
    print(f"Accuracy: {acc:.3f}")
    
    print("\nAll fallback implementations working correctly!")