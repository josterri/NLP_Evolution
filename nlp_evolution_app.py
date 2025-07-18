import streamlit as st
import logging
from utils import (
    handle_errors, 
    display_progress_sidebar, 
    initialize_session_state,
    ProgressTracker,
    check_dependencies
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import chapter controllers (temporarily disabling lazy loading)
from chapter0 import render_chapter_0
from chapter1 import render_chapter_1
from chapter2 import render_chapter_2
from chapter3 import render_chapter_3
from chapter4 import render_chapter_4
from chapter5 import render_chapter_5
from chapter6 import render_chapter_6
from chapter7 import render_chapter_7
from chapter8 import render_chapter_8
from chapter9 import render_chapter_9

# Import interactive modules
from quiz_system import render_quiz_interface
from code_exercises import render_code_exercise_interface
from datasets import render_dataset_explorer
from glossary import render_glossary
from theme_manager import apply_theme

# --- App Configuration ---
st.set_page_config(
    page_title="The Evolution of NLP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main App Logic ---
@handle_errors
def main():
    """Main function to run the Streamlit app."""
    # Initialize session state and check dependencies
    initialize_session_state()
    
    # Update current time for tracking
    import time
    st.session_state.current_time = time.time()
    
    if not check_dependencies():
        st.stop()
    
    # Progress tracking
    tracker = ProgressTracker()
    
    st.sidebar.title("📜 The Evolution of NLP")
    
    # Apply theme
    apply_theme()
    
    st.sidebar.markdown("---")
    
    # A dictionary for the new 10-chapter structure
    CHAPTERS = {
        "Chapter 0: Before Neural Networks": render_chapter_0,
        "Chapter 1: The Statistical Era": render_chapter_1,
        "Chapter 2: The Rise of Neural Networks & Embeddings": render_chapter_2,
        "Chapter 3: Sequential Models & The Power of Context": render_chapter_3,
        "Chapter 4: The Transformer Revolution": render_chapter_4,
        "Chapter 5: Applying the Foundations: Text Classification": render_chapter_5,
        "Chapter 6: The Rise of Generative Models": render_chapter_6,
        "Chapter 7: Build Your Own Generative Model": render_chapter_7,
        "Chapter 8: The Era of Large Language Models (LLMs)": render_chapter_8,
        "Chapter 9: Course Completion & Future Directions": render_chapter_9,
        "🧠 Knowledge Check Quizzes": render_quiz_interface,
        "💻 Interactive Code Exercises": render_code_exercise_interface,
        "📊 Dataset Explorer": render_dataset_explorer,
        "📚 NLP Glossary": render_glossary,
    }

    # Simple single radio button navigation
    selection = st.sidebar.radio("Go to:", list(CHAPTERS.keys()))
    
    # Track chapter visit
    tracker.mark_chapter_visited(selection)
    
    # Display progress in sidebar
    display_progress_sidebar()

    # Call the render function for the selected chapter
    # Each chapter's render function is now responsible for its own sub-navigation
    try:
        page = CHAPTERS[selection]
        page()
    except Exception as e:
        logger.error(f"Error rendering chapter {selection}: {e}")
        st.error(f"Error loading chapter content. Please try refreshing the page.")

    st.sidebar.markdown("---")
    st.sidebar.info("""
    This app demonstrates the key milestones in the evolution of Natural Language Processing,
    from simple statistical models to the foundations of modern AI.
    """)

if __name__ == "__main__":
    main()
