#!/usr/bin/env python3
"""
Test coding practice functionality in Chapter 1.
"""

import sys
import traceback

def test_coding_practice_imports():
    """Test if coding practice modules can be imported and basic functionality works."""
    print("Testing coding practice functionality...")
    
    try:
        # Test code exercises import
        from code_exercises import CodeExerciseManager, render_exercise_widget
        print("PASS: CodeExerciseManager imported successfully")
        
        # Test exercise manager initialization
        manager = CodeExerciseManager()
        print("PASS: CodeExerciseManager initialized successfully")
        
        # Test getting exercises for chapter1
        chapter1_exercises = manager.get_exercises_for_chapter("chapter1")
        if chapter1_exercises:
            print(f"PASS: Found exercises for chapter1: {len(chapter1_exercises.get('exercises', []))} exercises")
            
            # Print exercise details
            for exercise in chapter1_exercises.get('exercises', []):
                print(f"   - {exercise['title']}: {exercise['description']}")
        else:
            print("FAIL: No exercises found for chapter1")
            return False
        
        # Test session state initialization (basic check)
        print("PASS: Basic exercise system tests passed")
        return True
        
    except ImportError as e:
        print(f"FAIL: Import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Error testing coding practice: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_exercise_templates():
    """Test if exercise templates are properly formatted."""
    try:
        from code_exercises import CODE_EXERCISES
        
        if "chapter1" in CODE_EXERCISES:
            chapter1 = CODE_EXERCISES["chapter1"]
            print(f"PASS: Chapter 1 exercise template found: {chapter1['title']}")
            
            exercises = chapter1.get("exercises", [])
            for exercise in exercises:
                required_fields = ['id', 'title', 'description', 'template']
                missing_fields = [field for field in required_fields if field not in exercise]
                
                if missing_fields:
                    print(f"FAIL: Exercise {exercise.get('id', 'unknown')} missing fields: {missing_fields}")
                    return False
                else:
                    print(f"PASS: Exercise {exercise['id']} has all required fields")
            
            return True
        else:
            print("FAIL: No chapter1 exercises found in CODE_EXERCISES")
            return False
            
    except Exception as e:
        print(f"FAIL: Error testing exercise templates: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Coding Practice Functionality ===\n")
    
    test1_result = test_coding_practice_imports()
    print()
    
    test2_result = test_exercise_templates()
    print()
    
    overall_success = test1_result and test2_result
    
    print("=== Test Results ===")
    print(f"Import & Basic Functionality: {'PASS' if test1_result else 'FAIL'}")
    print(f"Exercise Templates: {'PASS' if test2_result else 'FAIL'}")
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    sys.exit(0 if overall_success else 1)