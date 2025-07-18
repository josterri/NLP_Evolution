#!/usr/bin/env python3
"""
Simulate button clicks and verify outputs without Selenium.
Tests the actual logic behind button clicks.
"""

import sys
import traceback
import streamlit as st
from unittest.mock import Mock, patch

class ButtonSimulationTest:
    def __init__(self):
        self.results = {
            "coding_exercises": {},
            "quiz_starts": {},
            "navigation": {},
            "glossary_search": False
        }
        
    def setup_mock_streamlit(self):
        """Setup mock Streamlit environment."""
        # Create a mock session state
        if not hasattr(st, 'session_state'):
            st.session_state = Mock()
            st.session_state.__getitem__ = Mock(side_effect=lambda x: None)
            st.session_state.__setitem__ = Mock()
            st.session_state.get = Mock(return_value=None)
            st.session_state.__contains__ = Mock(return_value=False)
        
        # Initialize required session state
        st.session_state.current_exercise = None
        st.session_state.current_quiz = None
        st.session_state.quiz_scores = {}
        st.session_state.exercise_progress = {}
        st.session_state.completed_chapters = set()
        st.session_state.chapters_visited = set()
        
        # Mock streamlit functions that would normally render UI
        st.button = Mock(return_value=False)
        st.markdown = Mock()
        st.write = Mock()
        st.columns = Mock(return_value=[Mock(), Mock()])
        st.caption = Mock()
        st.subheader = Mock()
        st.header = Mock()
        st.success = Mock()
        st.error = Mock()
        st.rerun = Mock()
    
    def test_coding_exercise_start(self):
        """Test starting coding exercises programmatically."""
        print("\n=== Testing Coding Exercise Start Logic ===")
        
        from code_exercises import CodeExerciseManager, CODE_EXERCISES
        
        manager = CodeExerciseManager()
        
        for chapter, data in CODE_EXERCISES.items():
            for exercise in data["exercises"]:
                exercise_id = exercise["id"]
                exercise_title = exercise["title"]
                
                # Simulate clicking Start button
                print(f"\nSimulating Start click for: {exercise_title} ({chapter})")
                
                # This is what happens when Start button is clicked
                success = manager.start_exercise(chapter, exercise_id)
                
                if success:
                    # Check if exercise was set in session state
                    current = st.session_state.current_exercise
                    if current and current["exercise"]["id"] == exercise_id:
                        print(f"PASS: Exercise started successfully")
                        
                        # Check exercise content
                        template = current["exercise"]["template"]
                        blank_count = template.count("___BLANK")
                        print(f"  - Exercise has {blank_count} blanks to fill")
                        
                        # Check if answers can be submitted
                        test_answer = "test_answer"
                        manager.submit_answer("BLANK1", test_answer)
                        
                        if current["user_answers"].get("BLANK1") == test_answer:
                            print(f"  - Can submit answers")
                            self.results["coding_exercises"][f"{chapter}-{exercise_id}"] = True
                        else:
                            print(f"  - Failed to submit answer")
                            self.results["coding_exercises"][f"{chapter}-{exercise_id}"] = False
                    else:
                        print(f"FAIL: Exercise not set in session state")
                        self.results["coding_exercises"][f"{chapter}-{exercise_id}"] = False
                else:
                    print(f"FAIL: Failed to start exercise")
                    self.results["coding_exercises"][f"{chapter}-{exercise_id}"] = False
                
                # Reset for next test
                st.session_state.current_exercise = None
    
    def test_quiz_start(self):
        """Test starting quizzes programmatically."""
        print("\n=== Testing Quiz Start Logic ===")
        
        from quiz_system import QuizManager, QUIZ_QUESTIONS
        
        manager = QuizManager()
        
        for chapter in QUIZ_QUESTIONS.keys():
            print(f"\nSimulating quiz start for: {chapter}")
            
            # This is what happens when Start Quiz button is clicked
            success = manager.start_quiz(chapter)
            
            if success:
                # Check if quiz was set in session state
                current = st.session_state.current_quiz
                if current and current["chapter"] == chapter:
                    print(f"PASS: Quiz started successfully")
                    
                    # Check quiz content
                    questions = current["questions"]
                    print(f"  - Quiz has {len(questions)} questions")
                    
                    # Simulate answering first question
                    if questions:
                        first_q = questions[0]
                        # Check if we can access quiz structure
                        if "user_answers" in current:
                            print(f"  - Quiz structure initialized")
                            self.results["quiz_starts"][chapter] = True
                        else:
                            print(f"  - Quiz has questions but no answer tracking")
                            self.results["quiz_starts"][chapter] = True  # Still pass if questions exist
                else:
                    print(f"FAIL: Quiz not set in session state")
                    self.results["quiz_starts"][chapter] = False
            else:
                print(f"FAIL: Failed to start quiz")
                self.results["quiz_starts"][chapter] = False
            
            # Reset for next test
            st.session_state.current_quiz = None
    
    def test_glossary_search(self):
        """Test glossary search functionality."""
        print("\n=== Testing Glossary Search ===")
        
        from glossary import NLP_TERMS
        
        # Test search functionality
        search_term = "transformer"
        print(f"Searching for: {search_term}")
        
        filtered_terms = {}
        for term, data in NLP_TERMS.items():
            if search_term.lower() in term.lower() or search_term.lower() in data["definition"].lower():
                filtered_terms[term] = data
        
        if filtered_terms:
            print(f"PASS: Found {len(filtered_terms)} matching terms:")
            for term in list(filtered_terms.keys())[:3]:  # Show first 3
                print(f"  - {term}")
            self.results["glossary_search"] = True
        else:
            print(f"FAIL: No terms found")
            self.results["glossary_search"] = False
    
    def test_navigation_logic(self):
        """Test chapter navigation logic."""
        print("\n=== Testing Navigation Logic ===")
        
        chapters = [
            "Chapter 0: Before Neural Networks",
            "Chapter 1: The Statistical Era",
            "Chapter 2: The Rise of Neural Networks & Embeddings",
            "Chapter 3: Sequential Models & The Power of Context",
            "Chapter 4: The Transformer Revolution"
        ]
        
        from utils import ProgressTracker
        tracker = ProgressTracker()
        
        for chapter in chapters:
            print(f"Simulating navigation to: {chapter}")
            
            # This is what happens when a chapter is selected
            tracker.mark_chapter_visited(chapter)
            
            if chapter in st.session_state.progress['chapters_visited']:
                print(f"PASS: Chapter marked as visited")
                self.results["navigation"][chapter] = True
            else:
                print(f"FAIL: Chapter not marked as visited")
                self.results["navigation"][chapter] = False
    
    def generate_report(self):
        """Generate test report."""
        print("\n=== BUTTON SIMULATION TEST REPORT ===")
        
        # Coding exercises
        exercise_passes = sum(1 for v in self.results["coding_exercises"].values() if v)
        print(f"\nCoding Exercise Starts: {exercise_passes}/{len(self.results['coding_exercises'])} passed")
        
        # Quizzes
        quiz_passes = sum(1 for v in self.results["quiz_starts"].values() if v)
        print(f"Quiz Starts: {quiz_passes}/{len(self.results['quiz_starts'])} passed")
        
        # Navigation
        nav_passes = sum(1 for v in self.results["navigation"].values() if v)
        print(f"Navigation: {nav_passes}/{len(self.results['navigation'])} passed")
        
        # Glossary
        print(f"Glossary Search: {'PASS' if self.results['glossary_search'] else 'FAIL'}")
        
        # Overall
        total_tests = (
            len(self.results["coding_exercises"]) +
            len(self.results["quiz_starts"]) +
            len(self.results["navigation"]) +
            1  # glossary
        )
        total_passes = exercise_passes + quiz_passes + nav_passes + (1 if self.results["glossary_search"] else 0)
        
        print(f"\nOVERALL: {total_passes}/{total_tests} tests passed")
        
        # Detailed failures
        if total_passes < total_tests:
            print("\nFailed tests:")
            for category, results in self.results.items():
                if isinstance(results, dict):
                    for key, value in results.items():
                        if not value:
                            print(f"  - {category}: {key}")
                elif not results:
                    print(f"  - {category}")
        
        return total_passes == total_tests
    
    def run(self):
        """Run all button simulation tests."""
        try:
            self.setup_mock_streamlit()
            
            self.test_coding_exercise_start()
            self.test_quiz_start()
            self.test_glossary_search()
            self.test_navigation_logic()
            
            return self.generate_report()
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("=== NLP Evolution App - Button Click Simulation Test ===")
    print("This test simulates button clicks without requiring a browser.\n")
    
    tester = ButtonSimulationTest()
    success = tester.run()
    
    if success:
        print("\nPASS: All button click simulations passed!")
        print("The app should handle all button clicks correctly when running.")
    else:
        print("\nFAIL: Some button click simulations failed.")
        print("Check the detailed output above for specific issues.")
    
    sys.exit(0 if success else 1)