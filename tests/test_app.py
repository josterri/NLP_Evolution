import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from nlp_evolution_app import main

@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    with patch('streamlit.sidebar.title') as mock_title, \
         patch('streamlit.sidebar.radio') as mock_radio, \
         patch('streamlit.sidebar.markdown') as mock_markdown, \
         patch('streamlit.sidebar.info') as mock_info:
        
        # Configure mock returns
        mock_radio.return_value = "Chapter 1: The Statistical Era"
        
        yield {
            'title': mock_title,
            'radio': mock_radio,
            'markdown': mock_markdown,
            'info': mock_info
        }

def test_app_sidebar_structure(mock_streamlit):
    """Test that the app's sidebar is structured correctly."""
    with patch('chapter1.render_chapter_1') as mock_render:
        main()
        
        # Check sidebar title
        mock_streamlit['title'].assert_called_once_with("📜 The Evolution of NLP")
        
        # Check chapter selection radio
        mock_streamlit['radio'].assert_called_once()
        radio_args = mock_streamlit['radio'].call_args[0]
        assert "Go to Chapter:" == radio_args[0]
        assert len(radio_args[1]) == 8  # 8 chapters
        
        # Check info section
        mock_streamlit['info'].assert_called_once()
        info_text = mock_streamlit['info'].call_args[0][0]
        assert "evolution of Natural Language Processing" in info_text

def test_chapter_navigation(mock_streamlit):
    """Test that selecting a chapter calls the correct render function."""
    with patch('chapter1.render_chapter_1') as mock_ch1, \
         patch('chapter2.render_chapter_2') as mock_ch2:
        
        # Test Chapter 1 selection
        mock_streamlit['radio'].return_value = "Chapter 1: The Statistical Era"
        main()
        mock_ch1.assert_called_once()
        mock_ch2.assert_not_called()
        
        # Reset mocks
        mock_ch1.reset_mock()
        mock_ch2.reset_mock()
        
        # Test Chapter 2 selection
        mock_streamlit['radio'].return_value = "Chapter 2: The Rise of Neural Networks & Embeddings"
        main()
        mock_ch1.assert_not_called()
        mock_ch2.assert_called_once()

@pytest.mark.integration
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