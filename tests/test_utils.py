"""
Unit tests for utility functions.
"""

import pytest
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    validate_input, 
    safe_file_operation, 
    format_number,
    ProgressTracker,
    handle_errors,
    initialize_session_state,
    log_user_interaction
)
import streamlit as st
from unittest.mock import Mock, patch, MagicMock

class TestValidateInput:
    """Test input validation functions."""
    
    def test_text_validation(self):
        """Test text input validation."""
        assert validate_input("hello", "text", min_length=1, max_length=10) == True
        assert validate_input("", "text", min_length=1) == False
        assert validate_input("a" * 1001, "text", max_length=1000) == False
        assert validate_input("valid text", "text", min_length=5, max_length=20) == True
    
    def test_number_validation(self):
        """Test number input validation."""
        assert validate_input("123", "number") == True
        assert validate_input("123.45", "number") == True
        assert validate_input("abc", "number") == False
        assert validate_input("", "number") == False
    
    def test_positive_number_validation(self):
        """Test positive number validation."""
        assert validate_input("123", "positive_number") == True
        assert validate_input("0", "positive_number") == False
        assert validate_input("-5", "positive_number") == False
        assert validate_input("123.45", "positive_number") == True

class TestSafeFileOperation:
    """Test safe file operation functions."""
    
    def test_file_exists(self):
        """Test file existence check."""
        with tempfile.NamedTemporaryFile() as tmp:
            assert safe_file_operation(tmp.name, "exists") == True
        
        assert safe_file_operation("nonexistent_file.txt", "exists") == False
    
    def test_file_readable(self):
        """Test file readability check."""
        with tempfile.NamedTemporaryFile() as tmp:
            assert safe_file_operation(tmp.name, "read") == True
        
        assert safe_file_operation("nonexistent_file.txt", "read") == False

class TestFormatNumber:
    """Test number formatting functions."""
    
    def test_format_large_numbers(self):
        """Test formatting of large numbers."""
        assert format_number(1000000) == "1.00M"
        assert format_number(1500000) == "1.50M"
        assert format_number(1000) == "1.00K"
        assert format_number(1500) == "1.50K"
        assert format_number(500) == "500.00"
    
    def test_format_decimal_places(self):
        """Test decimal places in formatting."""
        assert format_number(1500000, decimal_places=1) == "1.5M"
        assert format_number(1500000, decimal_places=3) == "1.500M"

class TestProgressTracker:
    """Test progress tracking functionality."""
    
    def test_progress_tracking(self):
        """Test basic progress tracking."""
        # Note: This test would need to be adapted for actual Streamlit session state
        # For now, we'll test the basic structure
        tracker = ProgressTracker()
        assert hasattr(tracker, 'mark_chapter_visited')
        assert hasattr(tracker, 'mark_section_completed')
        assert hasattr(tracker, 'mark_exercise_attempted')
        assert hasattr(tracker, 'get_progress_stats')

class TestHandleErrors:
    """Test the handle_errors decorator."""
    
    def test_successful_function(self):
        """Test that decorator doesn't interfere with successful execution."""
        @handle_errors
        def successful_func():
            return "success"
        
        assert successful_func() == "success"
    
    def test_function_with_exception(self):
        """Test that decorator handles exceptions gracefully."""
        @handle_errors
        def failing_func():
            raise ValueError("Test error")
        
        with patch('streamlit.error') as mock_error:
            result = failing_func()
            assert result is None
            mock_error.assert_called_once()


class TestSessionStateInitialization:
    """Test session state initialization."""
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_initialize_session_state(self, mock_session_state):
        """Test that session state is properly initialized."""
        initialize_session_state()
        
        assert 'progress_tracker' in mock_session_state
        assert 'current_chapter' in mock_session_state
        assert 'models_cache' in mock_session_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])