#!/usr/bin/env python3
"""
Test if the quiz interface fix works.
"""

import sys

def test_quiz_interface_load():
    """Test if quiz interface can load without errors."""
    print("Testing quiz interface load...")
    
    try:
        # Import the quiz module
        from quiz_system import render_quiz_interface, QuizManager
        print("PASS: Quiz module imported successfully")
        
        # Test QuizManager initialization
        manager = QuizManager()
        print("PASS: QuizManager initialized")
        
        # Test if session state initialization works
        manager.initialize_session_state()
        print("PASS: Session state initialization works")
        
        # Test available quizzes
        available = manager.get_available_quizzes()
        if available:
            print(f"PASS: Found {len(available)} available quizzes:")
            for chapter, title in list(available.items())[:3]:
                print(f"  - {chapter}: {title}")
        else:
            print("FAIL: No quizzes available")
            return False
        
        # Check if all required functions exist and have error handling
        from inspect import getsource
        
        # Check render_quiz_interface
        source = getsource(render_quiz_interface)
        if "@handle_errors" in source:
            print("PASS: render_quiz_interface has error handling")
        else:
            print("FAIL: render_quiz_interface missing error handling")
            return False
        
        print("\nQuiz interface should now load without errors!")
        return True
        
    except Exception as e:
        print(f"FAIL: Error testing quiz interface: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quiz_interface_load()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)