"""Test suite for quiz system module."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quiz_system import QuizManager, QUIZ_QUESTIONS
from unittest.mock import patch, MagicMock
import streamlit as st


class TestQuizManager:
    """Test QuizManager functionality."""
    
    def test_quiz_questions_structure(self):
        """Test that quiz questions are properly structured."""
        assert isinstance(QUIZ_QUESTIONS, dict)
        
        for chapter, chapter_data in QUIZ_QUESTIONS.items():
            assert 'title' in chapter_data
            assert 'questions' in chapter_data
            assert isinstance(chapter_data['questions'], list)
            
            for question in chapter_data['questions']:
                assert 'id' in question
                assert 'question' in question
                assert 'options' in question
                assert 'correct' in question
                assert 'explanation' in question
                assert isinstance(question['options'], list)
                assert isinstance(question['correct'], int)
                assert 0 <= question['correct'] < len(question['options'])
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_quiz_manager_initialization(self, mock_session_state):
        """Test QuizManager initialization."""
        manager = QuizManager()
        assert hasattr(manager, 'questions')
        assert hasattr(manager, 'initialize_session_state')
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_start_quiz(self, mock_session_state):
        """Test starting a quiz."""
        manager = QuizManager()
        
        # Test starting quiz for existing chapter
        result = manager.start_quiz('chapter1')
        assert result is True
        assert 'current_quiz' in mock_session_state
        assert mock_session_state['current_quiz']['chapter'] == 'chapter1'
        
        # Test starting quiz for non-existent chapter
        result = manager.start_quiz('chapter99')
        assert result is False
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_submit_answer(self, mock_session_state):
        """Test submitting quiz answers."""
        manager = QuizManager()
        
        # Start a quiz first
        manager.start_quiz('chapter1')
        
        # Submit an answer
        result = manager.submit_answer(0)  # First answer option
        assert result is True
        
        quiz = mock_session_state['current_quiz']
        assert len(quiz['user_answers']) == 1
        assert quiz['current_question'] == 1
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_quiz_completion(self, mock_session_state):
        """Test quiz completion and scoring."""
        manager = QuizManager()
        
        # Start quiz and complete all questions
        manager.start_quiz('chapter1')
        num_questions = len(manager.questions['chapter1']['questions'])
        
        for i in range(num_questions):
            manager.submit_answer(0)  # Answer with first option
        
        quiz = mock_session_state['current_quiz']
        assert quiz['completed'] is True
        assert 'end_time' in quiz
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_get_quiz_results(self, mock_session_state):
        """Test retrieving quiz results."""
        manager = QuizManager()
        
        # Test when no results exist
        results = manager.get_quiz_results('chapter1')
        assert results is None
        
        # Add some quiz scores
        mock_session_state['quiz_scores'] = {
            'chapter1': {
                'score': 80.0,
                'correct': 4,
                'total': 5
            }
        }
        
        results = manager.get_quiz_results('chapter1')
        assert results is not None
        assert results['score'] == 80.0
    
    def test_quiz_content_coverage(self):
        """Test that quiz questions cover expected chapters."""
        expected_chapters = ['chapter0', 'chapter1', 'chapter2', 'chapter3', 'chapter4']
        
        for chapter in expected_chapters:
            assert chapter in QUIZ_QUESTIONS, f"Missing quiz questions for {chapter}"
            assert len(QUIZ_QUESTIONS[chapter]['questions']) >= 3, f"Too few questions for {chapter}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])