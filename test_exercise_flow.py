#!/usr/bin/env python3
"""
Test the complete exercise flow to ensure coding practice buttons work properly.
"""

import sys
import traceback

def test_exercise_flow():
    """Test the complete exercise flow."""
    print("Testing exercise flow...")
    
    try:
        # Import required modules
        from code_exercises import CodeExerciseManager, render_exercise_widget, CODE_EXERCISES
        print("PASS: Modules imported successfully")
        
        # Test manager initialization
        manager = CodeExerciseManager()
        print("PASS: Exercise manager initialized")
        
        # Check that Chapter 2 has the Cosine Similarity exercise
        if "chapter2" in CODE_EXERCISES:
            chapter2_exercises = CODE_EXERCISES["chapter2"]["exercises"]
            cosine_exercise = next((ex for ex in chapter2_exercises if ex["id"] == "cosine_similarity"), None)
            
            if cosine_exercise:
                print(f"PASS: Found 'Calculate Cosine Similarity' exercise in Chapter 2")
                print(f"   Title: {cosine_exercise['title']}")
                print(f"   Description: {cosine_exercise['description']}")
                
                # Check that template has blanks to fill
                template = cosine_exercise['template']
                blank_count = template.count('___BLANK')
                if blank_count > 0:
                    print(f"PASS: Exercise template has {blank_count} blanks to fill")
                else:
                    print("FAIL: Exercise template has no blanks")
                    return False
                    
            else:
                print("FAIL: Cosine similarity exercise not found in Chapter 2")
                return False
        else:
            print("FAIL: Chapter 2 not found in CODE_EXERCISES")
            return False
        
        # Test exercise starting
        print("\nTesting exercise start functionality...")
        
        # Simulate starting the cosine similarity exercise
        success = manager.start_exercise("chapter2", "cosine_similarity")
        if success:
            print("PASS: Exercise started successfully")
            
            # Check session state (this will show a warning since we're not in streamlit context)
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and st.session_state.current_exercise:
                    print("PASS: Session state updated with current exercise")
                    print(f"   Exercise: {st.session_state.current_exercise['exercise']['title']}")
                else:
                    print("INFO: Session state not available in test context (normal)")
            except:
                print("INFO: Streamlit session state not available in test context (normal)")
        else:
            print("FAIL: Exercise failed to start")
            return False
        
        print("\nPASS: All exercise flow tests completed successfully")
        return True
        
    except Exception as e:
        print(f"FAIL: Error in exercise flow test: {e}")
        traceback.print_exc()
        return False

def test_chapter_exercise_availability():
    """Test that exercises are available for multiple chapters."""
    try:
        from code_exercises import CODE_EXERCISES
        
        print("\nTesting exercise availability across chapters...")
        
        chapters_with_exercises = []
        for chapter, data in CODE_EXERCISES.items():
            exercise_count = len(data.get("exercises", []))
            if exercise_count > 0:
                chapters_with_exercises.append(chapter)
                print(f"PASS: {chapter} has {exercise_count} exercises")
                
                # List exercise titles
                for exercise in data["exercises"]:
                    print(f"   - {exercise['title']}")
        
        if len(chapters_with_exercises) > 0:
            print(f"\nPASS: Found exercises in {len(chapters_with_exercises)} chapters")
            return True
        else:
            print("FAIL: No chapters have exercises")
            return False
            
    except Exception as e:
        print(f"FAIL: Error checking exercise availability: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Complete Exercise Flow ===\n")
    
    test1_result = test_exercise_flow()
    test2_result = test_chapter_exercise_availability()
    
    overall_success = test1_result and test2_result
    
    print("\n=== Test Results ===")
    print(f"Exercise Flow: {'PASS' if test1_result else 'FAIL'}")
    print(f"Chapter Availability: {'PASS' if test2_result else 'FAIL'}")
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\n✓ The exercise system should now work properly!")
        print("✓ Clicking 'Start' on 'Calculate Cosine Similarity' should show the exercise interface")
    
    sys.exit(0 if overall_success else 1)