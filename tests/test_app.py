import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

def test_app_title(app):
    """Test that the app title is correctly set."""
    app.run()
    assert app._config.page_title == "The Evolution of NLP"
    assert app._config.layout == "wide"
    assert app._config.initial_sidebar_state == "expanded"

def test_sidebar_title(app):
    """Test that the sidebar title is correctly displayed."""
    app.run()
    assert "📜 The Evolution of NLP" in app.sidebar

def test_chapter_navigation(app, chapter_names):
    """Test that all chapters are available in the navigation."""
    app.run()
    for chapter in chapter_names:
        assert chapter in app.sidebar

def test_chapter_selection(app, chapter_names):
    """Test that selecting each chapter works."""
    for chapter in chapter_names:
        app._mock.radio("Go to Chapter:", chapter)
        app.run()
        # Verify chapter header is present
        chapter_number = chapter.split(":")[0].strip()
        assert chapter_number in app._session_state

def test_chapter1_sections(app, chapter1_sections):
    """Test Chapter 1's section navigation."""
    app._mock.radio("Go to Chapter:", "Chapter 1: The Statistical Era")
    app.run()
    for section in chapter1_sections:
        assert section in app.sidebar

def test_sidebar_info(app):
    """Test that the sidebar info text is present."""
    app.run()
    assert "evolution of Natural Language Processing" in app.sidebar

@pytest.mark.parametrize("chapter_name,expected_header", [
    ("Chapter 1: The Statistical Era", "The Foundation - Predicting the Next Word"),
    ("Chapter 2: The Rise of Neural Networks & Embeddings", "Neural Networks & Embeddings"),
    ("Chapter 3: Sequential Models & The Power of Context", "Sequential Models"),
    ("Chapter 4: The Transformer Revolution", "The Transformer Revolution"),
    ("Chapter 5: Applying the Foundations: Text Classification", "Text Classification"),
    ("Chapter 6: The Rise of Generative Models", "Generative Models"),
    ("Chapter 7: Build Your Own Generative Model", "Build Your Own Model"),
    ("Chapter 8: The Era of Large Language Models (LLMs)", "Large Language Models"),
])
def test_chapter_headers(app, chapter_name, expected_header):
    """Test that each chapter displays the correct header."""
    app._mock.radio("Go to Chapter:", chapter_name)
    app.run()
    assert expected_header in app._session_state

def test_app_error_handling(app):
    """Test that the app handles errors gracefully."""
    # Test with invalid chapter selection
    with pytest.raises(KeyError):
        app._mock.radio("Go to Chapter:", "Invalid Chapter")
        app.run()

def test_session_state_persistence(app, chapter_names):
    """Test that session state persists between interactions."""
    # Select first chapter
    app._mock.radio("Go to Chapter:", chapter_names[0])
    app.run()
    initial_state = app._session_state.copy()
    
    # Select second chapter
    app._mock.radio("Go to Chapter:", chapter_names[1])
    app.run()
    # Verify state changed
    assert app._session_state != initial_state

def test_markdown_rendering(app):
    """Test that markdown content is properly rendered."""
    app.run()
    # Check for common markdown elements
    assert "---" in app._session_state  # Horizontal rule
    assert "#" in app._session_state    # Headers 