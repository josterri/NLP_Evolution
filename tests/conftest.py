import pytest
from streamlit.testing.v1 import AppTest

@pytest.fixture
def app():
    """Fixture that provides a Streamlit test app instance."""
    return AppTest.from_file("nlp_evolution_app.py")

@pytest.fixture
def chapter_names():
    """Fixture that provides the list of chapter names."""
    return [
        "Chapter 1: The Statistical Era",
        "Chapter 2: The Rise of Neural Networks & Embeddings",
        "Chapter 3: Sequential Models & The Power of Context",
        "Chapter 4: The Transformer Revolution",
        "Chapter 5: Applying the Foundations: Text Classification",
        "Chapter 6: The Rise of Generative Models",
        "Chapter 7: Build Your Own Generative Model",
        "Chapter 8: The Era of Large Language Models (LLMs)",
    ]

@pytest.fixture
def chapter1_sections():
    """Fixture that provides Chapter 1 section names."""
    return [
        "1.1: N-grams & The Interactive Demo",
        "1.2: The Sparsity Problem",
        "1.3: Smoothing Techniques",
    ] 