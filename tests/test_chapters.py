import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

def test_chapter1_structure(app):
    """Test Chapter 1's structure and navigation."""
    app._mock.radio("Go to Chapter:", "Chapter 1: The Statistical Era")
    app.run()
    
    # Check header
    assert "The Foundation - Predicting the Next Word" in app._session_state
    
    # Check sections
    sections = [
        "1.1: N-grams & The Interactive Demo",
        "1.2: The Sparsity Problem",
        "1.3: Smoothing Techniques"
    ]
    for section in sections:
        assert section in app.sidebar

def test_chapter_imports():
    """Test that all chapter modules can be imported."""
    import chapter1
    import chapter2
    import chapter3
    import chapter4
    import chapter5
    import chapter6
    import chapter7
    import chapter8
    
    # Verify each chapter has the required render function
    assert hasattr(chapter1, 'render_chapter_1')
    assert hasattr(chapter2, 'render_chapter_2')
    assert hasattr(chapter3, 'render_chapter_3')
    assert hasattr(chapter4, 'render_chapter_4')
    assert hasattr(chapter5, 'render_chapter_5')
    assert hasattr(chapter6, 'render_chapter_6')
    assert hasattr(chapter7, 'render_chapter_7')
    assert hasattr(chapter8, 'render_chapter_8')

def test_chapter_submodules():
    """Test that chapter submodules can be imported."""
    # Test Chapter 1 submodules
    from chapter1_1 import render_1_1
    from chapter1_2 import render_1_2
    from chapter1_3 import render_1_3
    
    # Verify functions exist
    assert callable(render_1_1)
    assert callable(render_1_2)
    assert callable(render_1_3)

def test_chapter1_navigation(app):
    """Test Chapter 1's section navigation."""
    app._mock.radio("Go to Chapter:", "Chapter 1: The Statistical Era")
    app.run()
    
    # Test each section
    sections = [
        "1.1: N-grams & The Interactive Demo",
        "1.2: The Sparsity Problem",
        "1.3: Smoothing Techniques"
    ]
    
    for section in sections:
        app._mock.radio("Chapter 1 Sections:", section, key="ch1_nav")
        app.run()
        # Verify section content is displayed
        section_number = section.split(":")[0].strip()
        assert section_number in app._session_state

@pytest.mark.integration
def test_chapter_dependencies():
    """Test that all required dependencies for chapters are available."""
    import nltk
    import numpy
    import pandas
    import torch
    import plotly
    import streamlit
    
    # Verify NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/brown')
    except LookupError as e:
        pytest.fail(f"Required NLTK data not found: {e}")

def test_chapter_content_rendering(app):
    """Test that each chapter renders its content correctly."""
    app._mock.radio("Go to Chapter:", "Chapter 1: The Statistical Era")
    app.run()
    
    # Check for common content elements
    assert "---" in app._session_state  # Horizontal rule
    assert "Chapter 1" in app._session_state  # Chapter title
    
    # Test markdown formatting
    assert "#" in app._session_state  # Headers
    assert "*" in app._session_state or "_" in app._session_state  # Emphasis

def test_chapter_state_isolation(app):
    """Test that each chapter maintains its own state."""
    # Set state in Chapter 1
    app._mock.radio("Go to Chapter:", "Chapter 1: The Statistical Era")
    app._mock.radio("Chapter 1 Sections:", "1.1: N-grams & The Interactive Demo", key="ch1_nav")
    app.run()
    ch1_state = app._session_state.copy()
    
    # Switch to Chapter 2
    app._mock.radio("Go to Chapter:", "Chapter 2: The Rise of Neural Networks & Embeddings")
    app.run()
    ch2_state = app._session_state.copy()
    
    # Verify states are different
    assert ch1_state != ch2_state 