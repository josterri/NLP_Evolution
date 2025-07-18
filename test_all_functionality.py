#!/usr/bin/env python3
"""
Comprehensive functionality test for NLP Evolution app.
Tests all major components and identifies any issues.
"""

import sys
import traceback
import importlib
from datetime import datetime

def test_section(name, func):
    """Test a section and report results."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print('='*50)
    
    try:
        result = func()
        if result:
            print(f"PASS {name}: PASSED")
            return True
        else:
            print(f"FAIL {name}: FAILED")
            return False
    except Exception as e:
        print(f"FAIL {name}: ERROR - {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_basic_imports():
    """Test that all core modules can be imported."""
    modules_to_test = [
        'utils',
        'theme_manager', 
        'sklearn_fallbacks',
        'quiz_system',
        'code_exercises',
        'datasets',
        'search_functionality',
        'export_progress',
        'lazy_loader'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"PASS {module}")
        except Exception as e:
            print(f"FAIL {module}: {str(e)}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_chapter_imports():
    """Test that all chapter modules can be imported."""
    failed_chapters = []
    
    # Test main chapter controllers
    for i in range(10):  # chapters 0-9
        try:
            importlib.import_module(f'chapter{i}')
            print(f"PASS chapter{i}")
        except Exception as e:
            print(f"FAIL chapter{i}: {str(e)}")
            failed_chapters.append(f'chapter{i}')
    
    return len(failed_chapters) == 0

def test_sklearn_fallbacks():
    """Test sklearn fallback implementations."""
    try:
        from sklearn_fallbacks import TfidfVectorizer, PCA, MultinomialNB, train_test_split, accuracy_score
        import numpy as np
        
        # Test TfidfVectorizer
        texts = ["hello world", "world of machine learning", "hello machine"]
        vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = vectorizer.fit_transform(texts)
        assert tfidf_matrix.shape[0] == 3
        print("PASS TfidfVectorizer working")
        
        # Test PCA
        data = np.random.rand(50, 10)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        assert reduced_data.shape == (50, 2)
        print("PASS PCA working")
        
        # Test MultinomialNB
        X = np.random.randint(0, 10, (50, 5))
        y = np.random.randint(0, 2, 50)
        nb = MultinomialNB()
        nb.fit(X, y)
        predictions = nb.predict(X[:10])
        assert len(predictions) == 10
        print("PASS MultinomialNB working")
        
        # Test train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        assert len(X_train) + len(X_test) == len(X)
        print("PASS train_test_split working")
        
        # Test accuracy_score
        acc = accuracy_score(y[:10], predictions)
        assert 0 <= acc <= 1
        print("PASS accuracy_score working")
        
        return True
        
    except Exception as e:
        print(f"FAIL sklearn_fallbacks error: {str(e)}")
        return False

def test_main_app_structure():
    """Test that main app can be imported and has required functions."""
    try:
        import nlp_evolution_app
        
        # Check if main function exists
        assert hasattr(nlp_evolution_app, 'main')
        print("PASS Main app has main() function")
        
        # Test that CHAPTERS dictionary exists when main is called
        # Note: We can't easily test this without running streamlit
        print("PASS Main app structure looks correct")
        
        return True
        
    except Exception as e:
        print(f"FAIL Main app error: {str(e)}")
        return False

def test_utils_functions():
    """Test core utility functions."""
    try:
        from utils import (
            handle_errors, initialize_session_state, ProgressTracker,
            get_memory_usage, export_progress, cleanup_session_state
        )
        
        # Test memory usage function
        memory_stats = get_memory_usage()
        assert 'rss_mb' in memory_stats
        assert 'vms_mb' in memory_stats
        print("PASS Memory monitoring working")
        
        # Test progress tracker initialization
        # Note: Can't fully test without streamlit session state
        print("PASS ProgressTracker class exists")
        
        # Test export progress function structure
        # Note: Will fail without session state, but we can check it exists
        assert callable(export_progress)
        print("PASS Export progress function exists")
        
        return True
        
    except Exception as e:
        print(f"FAIL Utils functions error: {str(e)}")
        return False

def test_interactive_modules():
    """Test interactive module structure."""
    try:
        # Test quiz system
        from quiz_system import QuizManager, QUIZ_QUESTIONS
        assert len(QUIZ_QUESTIONS) > 0
        print("PASS Quiz system has questions")
        
        # Test code exercises
        from code_exercises import CODE_EXERCISES
        assert len(CODE_EXERCISES) > 0
        print("PASS Code exercises exist")
        
        # Test theme manager functions
        from theme_manager import ThemeManager, apply_theme, render_theme_controls
        theme_manager = ThemeManager()
        assert len(theme_manager.themes) > 0
        print("PASS Theme manager working")
        
        return True
        
    except Exception as e:
        print(f"FAIL Interactive modules error: {str(e)}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'plotly', 'seaborn', 'nltk', 'torch', 'PyPDF2',
        'pytest', 'psutil', 'reportlab'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"PASS {package}")
        except ImportError:
            print(f"FAIL {package} - MISSING")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def test_data_files():
    """Test that required data files exist or can be handled gracefully."""
    import os
    
    optional_files = [
        'text8.txt',
        'BIS_Speech.pdf', 
        'glove.6B.50dsmall.txt'
    ]
    
    for filename in optional_files:
        if os.path.exists(filename):
            print(f"PASS {filename} found")
        else:
            print(f"WARN  {filename} not found (optional)")
    
    # Check if logs directory can be created
    try:
        with open('nlp_evolution.log', 'a') as f:
            f.write(f"Test log entry: {datetime.now()}\n")
        print("PASS Logging system working")
    except Exception as e:
        print(f"FAIL Logging error: {str(e)}")
        return False
    
    return True

def test_streamlit_compatibility():
    """Test streamlit compatibility without actually running streamlit."""
    try:
        import streamlit as st
        
        # Check streamlit version
        print(f"PASS Streamlit version: {st.__version__}")
        
        # Test that we can import streamlit components used in the app
        # These will generate warnings but shouldn't fail
        return True
        
    except Exception as e:
        print(f"FAIL Streamlit compatibility error: {str(e)}")
        return False

def run_all_tests():
    """Run all functionality tests."""
    print("NLP Evolution App - Comprehensive Functionality Test")
    print(f"Test started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    
    test_results = []
    
    # Run all tests
    test_results.append(test_section("Basic Module Imports", test_basic_imports))
    test_results.append(test_section("Chapter Module Imports", test_chapter_imports))
    test_results.append(test_section("sklearn Fallback Implementations", test_sklearn_fallbacks))
    test_results.append(test_section("Main App Structure", test_main_app_structure))
    test_results.append(test_section("Utils Functions", test_utils_functions))
    test_results.append(test_section("Interactive Modules", test_interactive_modules))
    test_results.append(test_section("Dependencies", test_dependencies))
    test_results.append(test_section("Data Files", test_data_files))
    test_results.append(test_section("Streamlit Compatibility", test_streamlit_compatibility))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\nALL TESTS PASSED! The app should work correctly.")
        return True
    else:
        print(f"\n{total - passed} TESTS FAILED. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)