import pytest
from chapter1_1 import build_ngram_model, get_word_counts_for_context
from collections import Counter

@pytest.fixture
def sample_text():
    return "the cat sat on the mat. the cat was happy."

@pytest.fixture
def bigram_model(sample_text):
    model, tokens = build_ngram_model(sample_text, n=2)
    return model, tokens

@pytest.fixture
def trigram_model(sample_text):
    model, tokens = build_ngram_model(sample_text, n=3)
    return model, tokens

def test_build_ngram_model_empty():
    """Test that empty text returns None."""
    model, tokens = build_ngram_model("", n=2)
    assert model is None

def test_build_ngram_model_short():
    """Test that text shorter than n returns None."""
    model, tokens = build_ngram_model("the cat", n=3)
    assert model is None

def test_build_bigram_model(bigram_model):
    """Test bigram model construction."""
    model, tokens = bigram_model
    
    # Test model structure
    assert isinstance(model, dict)
    assert all(isinstance(k, tuple) for k in model.keys())
    assert all(isinstance(v, Counter) for v in model.values())
    
    # Test specific bigram predictions
    the_context = tuple(['the'])
    cat_counts = model[the_context]
    assert cat_counts['cat'] == 2  # "the cat" appears twice
    assert cat_counts['mat'] == 1  # "the mat" appears once

def test_build_trigram_model(trigram_model):
    """Test trigram model construction."""
    model, tokens = trigram_model
    
    # Test model structure
    assert isinstance(model, dict)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in model.keys())
    assert all(isinstance(v, Counter) for v in model.values())
    
    # Test specific trigram predictions
    the_cat_context = tuple(['the', 'cat'])
    assert model[the_cat_context]['sat'] == 1
    assert model[the_cat_context]['was'] == 1

def test_text_preprocessing(bigram_model):
    """Test text preprocessing (lowercase, punctuation removal)."""
    model, tokens = bigram_model
    
    # Check tokens are lowercase
    assert all(token.islower() for token in tokens)
    # Check punctuation was removed
    assert all(not any(c in token for c in '.,!?') for token in tokens)

def test_get_word_counts():
    """Test word count retrieval for a context."""
    text = "the cat sat on the mat"
    model, _ = build_ngram_model(text, n=2)
    
    # Test existing context
    counts = get_word_counts_for_context(model, tuple(['the']))
    assert counts['cat'] == 1
    assert counts['mat'] == 1
    
    # Test non-existent context
    counts = get_word_counts_for_context(model, tuple(['dog']))
    assert counts is None 