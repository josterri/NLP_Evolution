import streamlit as st

# Import the render functions from the new sub-chapter files
from chapter1_1 import render_1_1
from chapter1_2 import render_1_2
from chapter1_3 import render_1_3

# Import interactive modules
from code_exercises import render_exercise_widget
from utils import show_progress, monitor_memory_usage

def render_chapter_1():
    """Renders all content for Chapter 1 by controlling sub-chapters."""
    # Monitor memory usage
    memory_stats = monitor_memory_usage()
    
    st.header("Chapter 1: The Foundation - Predicting the Next Word")
    
    # Show chapter progress
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Memory Usage", f"{memory_stats['rss_mb']:.0f}MB")
        with col2:
            completed_sections = len([s for s in ['1.1', '1.2', '1.3'] if st.session_state.get(f'ch1_{s}_completed', False)])
            st.metric("Sections Completed", f"{completed_sections}/3")
        with col3:
            if 'chapter1_start_time' not in st.session_state:
                import time
                st.session_state.chapter1_start_time = st.session_state.get('app_start_time', time.time())
            current_time = st.session_state.get('current_time', st.session_state.get('app_start_time', 0))
            time_spent = max(0, (current_time - st.session_state.chapter1_start_time) / 60)
            st.metric("Time in Chapter", f"{time_spent:.1f}min")

    SUB_CHAPTERS = {
        "1.1: N-grams & The Interactive Demo": render_1_1,
        "1.2: The Sparsity Problem": render_1_2,
        "1.3: Smoothing Techniques": render_1_3,
    }

    sub_selection = st.sidebar.radio(
        "Chapter 1 Sections:",
        list(SUB_CHAPTERS.keys()),
        key="ch1_nav"
    )
    
    st.markdown("---")

    # Call the render function for the selected page with progress indicator
    with show_progress(f"Loading {sub_selection}...") as progress:
        progress(0.3, "Preparing content...")
        page = SUB_CHAPTERS[sub_selection]
        progress(0.7, "Rendering interface...")
        page()
        progress(1.0, "Complete!")
    
    # Add coding practice at the end of the chapter
    st.markdown("---")
    render_exercise_widget("chapter1")
