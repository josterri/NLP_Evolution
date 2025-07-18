"""Test suite for code exercises module."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_exercises import CodeExerciseManager, CODE_EXERCISES
from unittest.mock import patch, MagicMock
import streamlit as st


class TestCodeExerciseManager:
    """Test CodeExerciseManager functionality."""
    
    def test_code_exercises_structure(self):
        """Test that code exercises are properly structured."""
        assert isinstance(CODE_EXERCISES, dict)
        
        for chapter, chapter_data in CODE_EXERCISES.items():
            assert 'title' in chapter_data
            assert 'exercises' in chapter_data
            assert isinstance(chapter_data['exercises'], list)
            
            for exercise in chapter_data['exercises']:
                assert 'id' in exercise
                assert 'title' in exercise
                assert 'description' in exercise
                assert 'template' in exercise
                assert 'blanks' in exercise
                assert isinstance(exercise['blanks'], list)
                
                for blank in exercise['blanks']:
                    assert 'blank' in blank
                    assert 'correct' in blank
                    assert 'explanation' in blank
                    if 'options' in blank:
                        assert isinstance(blank['options'], list)
                        assert blank['correct'] in blank['options']
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_exercise_manager_initialization(self, mock_session_state):
        """Test CodeExerciseManager initialization."""
        manager = CodeExerciseManager()
        assert hasattr(manager, 'exercises')
        assert hasattr(manager, 'initialize_session_state')
        assert hasattr(manager, 'start_exercise')
        assert hasattr(manager, 'submit_answer')
        assert hasattr(manager, 'check_answers')
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_start_exercise(self, mock_session_state):
        """Test starting an exercise."""
        manager = CodeExerciseManager()
        
        # Test starting exercise for existing chapter
        result = manager.start_exercise('chapter1', 'ngram_basic')
        assert result is True
        assert 'current_exercise' in mock_session_state
        assert mock_session_state['current_exercise']['chapter'] == 'chapter1'
        
        # Test starting exercise for non-existent chapter
        result = manager.start_exercise('chapter99', 'test')
        assert result is False
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_submit_answer(self, mock_session_state):
        """Test submitting exercise answers."""
        manager = CodeExerciseManager()
        
        # Start an exercise first
        manager.start_exercise('chapter1', 'ngram_basic')
        
        # Submit an answer
        result = manager.submit_answer('___BLANK1___', 'words[i]')
        assert result is True
        
        exercise = mock_session_state['current_exercise']
        assert '___BLANK1___' in exercise['user_answers']
        assert exercise['user_answers']['___BLANK1___'] == 'words[i]'
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_check_answers(self, mock_session_state):
        """Test checking exercise answers."""
        manager = CodeExerciseManager()
        
        # Start exercise and submit some answers
        manager.start_exercise('chapter1', 'ngram_basic')
        
        # Get the first exercise
        exercise_data = CODE_EXERCISES['chapter1']['exercises'][0]
        
        # Submit correct answers
        for blank in exercise_data['blanks']:
            manager.submit_answer(blank['blank'], blank['correct'])
        
        # Check answers
        results = manager.check_answers()
        assert results is not None
        assert 'score' in results
        assert 'correct' in results
        assert 'total' in results
        assert results['score'] == 100.0  # All answers correct
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_get_completed_code(self, mock_session_state):
        """Test getting completed code with user answers."""
        manager = CodeExerciseManager()
        
        # Start exercise
        manager.start_exercise('chapter1', 'ngram_basic')
        
        # Submit some answers
        manager.submit_answer('___BLANK1___', 'words[i]')
        manager.submit_answer('___BLANK2___', 'words[i+1]')
        
        # Get completed code
        completed_code = manager.get_completed_code()
        assert '___BLANK1___' not in completed_code
        assert '___BLANK2___' not in completed_code
        assert 'words[i]' in completed_code
        assert 'words[i+1]' in completed_code
    
    def test_exercise_content_coverage(self):
        """Test that exercises cover expected chapters."""
        expected_chapters = ['chapter1', 'chapter2', 'chapter3']
        
        for chapter in expected_chapters:
            assert chapter in CODE_EXERCISES, f"Missing code exercises for {chapter}"
            assert len(CODE_EXERCISES[chapter]['exercises']) >= 1, f"No exercises for {chapter}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])