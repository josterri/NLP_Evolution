# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
# Recommended: Auto-find port and open browser
python launch_app.py

# Alternative: Standard Streamlit
streamlit run nlp_evolution_app.py
```

### Testing
```bash
# Run comprehensive end-to-end tests
python test_e2e.py                    # Basic functionality test
python test_e2e.py --full             # Full Selenium browser test

# Run button simulation tests
python test_button_simulation.py      # Test all interactive elements

# Run specific functionality tests
python test_all_functionality.py      # Core functionality verification
python test_startup.py               # App startup verification
python test_chapter1.py              # Chapter 1 specific tests

# Run pytest-based unit tests
pytest tests/                         # All unit tests
pytest tests/test_utils.py           # Utility function tests
pytest tests/test_quiz_system.py     # Quiz system tests
pytest tests/test_code_exercises.py  # Code exercise tests
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Download required NLTK data (first time setup)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Critical Compatibility Note:** This project uses custom sklearn fallbacks instead of scikit-learn due to Python 3.13 compatibility issues. The fallback implementations are in `sklearn_fallbacks.py` and provide educational implementations of TfidfVectorizer, PCA, MultinomialNB, train_test_split, and accuracy_score.

## Architecture Overview

This is an educational Streamlit application teaching NLP evolution through 10 interactive chapters (0-9). The architecture follows a hierarchical modular design with sophisticated interactive features.

### Core Application Structure
- **Main Entry**: `nlp_evolution_app.py` - Central orchestrator with navigation system
- **Chapter Controllers**: `chapter0.py` through `chapter9.py` - Handle chapter-level navigation and routing
- **Section Modules**: `chapter{N}_{section}.py` - Individual learning sections (e.g., `chapter2_6.py`)
- **Core Utilities**: `utils.py` - Error handling, progress tracking, session management, metrics calculation

### Interactive Module Ecosystem
The app features a comprehensive interactive learning system:

- **Quiz System** (`quiz_system.py`): Full quiz management with scoring, progress tracking, and detailed feedback
- **Code Exercises** (`code_exercises.py`): Fill-in-the-blank coding exercises with validation and immediate feedback
- **Dataset Explorer** (`datasets.py`): Interactive data exploration and visualization
- **Glossary** (`glossary.py`): Comprehensive NLP term definitions with search and filtering
- **Theme Manager** (`theme_manager.py`): Light/dark themes with accessibility options

**Note:** Recent architectural changes removed search functionality and export progress features to focus on core learning experience.

### Custom ML Implementation Layer
**Critical Architecture Decision**: Due to Python 3.13 compatibility issues, the app uses custom educational ML implementations:

- **sklearn_fallbacks.py**: Complete replacement for scikit-learn with pedagogical focus
  - TfidfVectorizer with proper IDF calculation and stop word handling
  - PCA with eigenvalue decomposition and explained variance
  - MultinomialNB with Laplace smoothing
  - Utility functions (train_test_split, accuracy_score)

### Session State Management Pattern
The app uses a sophisticated session state architecture:
- **Progress Tracking**: `ProgressTracker` class with JSON persistence
- **Memory Management**: Automatic cleanup with `cleanup_session_state()`
- **Theme Persistence**: Theme state management across page reloads
- **Quiz/Exercise State**: Detailed progress tracking for interactive elements

### Chapter Content Organization
- **Chapter 0**: Pre-neural era (rule-based systems, TF-IDF)
- **Chapters 1-3**: Statistical models to embeddings
- **Chapter 4**: Transformer architecture deep dive
- **Chapters 5-7**: Practical applications and implementations
- **Chapters 8-9**: Modern LLMs and course completion

### Advanced Features Integration
- **Lazy Loading System** (`lazy_loader.py`): Memory-efficient module loading
- **Memory Monitoring**: Real-time memory usage tracking with psutil
- **Help System**: Context-sensitive help with keyboard shortcuts
- **Mobile Responsive**: Custom CSS for mobile devices

## Critical Implementation Patterns

### Error Handling Architecture
Always use the `@handle_errors` decorator for user-facing functions:
```python
@handle_errors
def your_function():
    # Function implementation with automatic error handling
```

### Navigation Pattern
Use `st.rerun()` (not deprecated `st.experimental_rerun()`) for page refreshes. The app handles navigation through session state with automatic cleanup.

### Coding Exercise Widget Pattern
The `render_exercise_widget()` function follows a dual-state pattern:
```python
def render_exercise_widget(chapter: str):
    # Check if there's an active exercise from this chapter
    if (st.session_state.current_exercise and 
        st.session_state.current_exercise.get("chapter") == chapter):
        # Render the active exercise interface
        render_active_exercise(exercise_manager)
    else:
        # Show available exercises with Start buttons
        # Display exercise list with descriptions
```

### Session State Initialization Pattern
Critical for avoiding 'st' variable access errors:
```python
def initialize_session_state():
    # Always initialize in functions, not at module level
    if 'current_exercise' not in st.session_state:
        st.session_state.current_exercise = None
```

### Progress Context Manager Pattern
Use the `show_progress` context manager for loading operations:
```python
with show_progress(f"Loading {section}...") as progress:
    progress(0.3, "Preparing content...")
    # Do work
    progress(1.0, "Complete!")
```

### Memory Management Protocol
The app implements sophisticated memory management:
- Automatic cleanup of large objects from session state
- Memory usage monitoring with configurable thresholds
- Lazy loading for modules to reduce initial memory footprint

## Development Workflow

### Adding New Interactive Features
1. Create module in root directory (e.g., `new_feature.py`)
2. Add render function following naming convention
3. Import in `nlp_evolution_app.py` 
4. Add to CHAPTERS dictionary
5. Update progress tracking if needed
6. Add tests in `tests/test_new_feature.py`

### Working with Custom ML Implementations
When adding ML functionality:
1. Use imports from `sklearn_fallbacks.py`
2. Ensure educational clarity in implementations
3. Add proper docstrings explaining algorithms
4. Test with various input sizes
5. Include fallback error handling

### Chapter Section Development
1. Create `chapter{N}_{section}.py` with `render_{N}_{section}()` function
2. Import in corresponding `chapter{N}.py` controller
3. Add to chapter navigation SUB_CHAPTERS dictionary
4. Update progress tracking configuration
5. Add quiz questions in `quiz_system.py` if applicable
6. Add code exercises in `code_exercises.py` if applicable

## Testing Strategy

### Test Structure
```bash
tests/
├── test_utils.py           # Core utilities and error handling
├── test_quiz_system.py     # Quiz functionality
├── test_code_exercises.py  # Code exercise system
└── installs.py            # Test setup utilities
```

### Critical Test Areas
- Error handling decorator functionality
- Session state management
- Progress tracking persistence
- Theme switching
- Memory cleanup operations
- Custom ML algorithm correctness

## Data and Resources

### Required Data Files
- `text8.txt`: Training corpus for word embeddings
- `BIS_Speech.pdf`: Sample document for PDF processing demos
- `glove.6B.50dsmall.txt`: Pre-trained GloVe embeddings (optional)

### NLTK Setup
Essential downloads for full functionality:
```python
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
```

## Performance Considerations

### Caching Strategy
- Use `@st.cache_data` for expensive computations
- Implement lazy loading for large datasets
- Cache model training results in session state
- Monitor memory usage with `get_memory_usage()`

### Memory Management
- Clear large objects when switching chapters
- Use generators for large text processing
- Implement configurable memory thresholds
- Automatic cleanup when memory limits exceeded

## Debugging and Troubleshooting

### Common Issues
- **sklearn Import Errors**: Use fallback implementations from `sklearn_fallbacks.py`
- **Session State Errors**: Check `initialize_session_state()` call in main app
- **'st' Variable Access Errors**: Ensure session state is initialized in functions, not at module level
- **Coding Exercise Buttons Not Working**: Verify `render_exercise_widget()` follows dual-state pattern
- **Memory Issues**: Enable memory monitoring and cleanup
- **Unicode Encoding in Logs**: Common with emoji characters, doesn't affect functionality

### Debug Tools
```bash
# View application logs
tail -f nlp_evolution.log

# Monitor memory usage
python -c "from utils import get_memory_usage; print(get_memory_usage())"

# Test sklearn fallbacks
python sklearn_fallbacks.py

# Test interactive elements
python test_button_simulation.py

# Verify app startup
python test_startup.py
```

### Testing Status
Current test results show:
- **Button Simulation**: 26/26 tests passing ✅
- **Core Functionality**: 7/9 tests passing ✅
- **Interactive Elements**: All working correctly ✅
- **Known Issues**: 2 runtime errors in Chapter 2 and quiz system need fixing

## Security and Best Practices

### Input Validation
- Use `validate_input()` for all user inputs
- Implement proper type checking
- Sanitize file paths with `safe_file_operation()`

### Session Management
- Initialize all session state keys through `initialize_session_state()`
- Implement proper cleanup to prevent memory leaks
- Use defensive programming for session state access

### Error Handling
- Never expose sensitive system information in error messages
- Log all errors with appropriate detail levels
- Provide user-friendly error messages through Streamlit UI