#!/usr/bin/env python3
"""
End-to-end test for NLP Evolution app.
Tests all buttons and verifies correct output is generated.
"""

import subprocess
import time
import sys
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
import signal

class NLPEvolutionE2ETest:
    def __init__(self, headless=True):
        self.headless = headless
        self.app_process = None
        self.driver = None
        self.app_url = None
        self.results = {
            "chapters": {},
            "coding_exercises": {},
            "quizzes": {},
            "glossary": False,
            "dataset_explorer": False
        }
        
    def start_app(self):
        """Start the Streamlit app and wait for it to be ready."""
        print("Starting NLP Evolution app...")
        
        # Find a free port
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        
        self.app_url = f"http://localhost:{port}"
        
        # Start the app
        self.app_process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'nlp_evolution_app.py', 
             '--server.port', str(port), '--server.headless', 'true'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if sys.platform != 'win32' else None
        )
        
        # Wait for app to start
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(self.app_url)
                if response.status_code == 200:
                    print(f"App started successfully at {self.app_url}")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("Failed to start app")
        return False
    
    def setup_driver(self):
        """Setup Selenium WebDriver."""
        print("Setting up WebDriver...")
        
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.get(self.app_url)
            # Wait for Streamlit to load
            time.sleep(5)
            print("WebDriver setup complete")
            return True
        except Exception as e:
            print(f"Failed to setup WebDriver: {e}")
            print("Make sure Chrome and ChromeDriver are installed")
            return False
    
    def test_chapter_navigation(self):
        """Test navigation through all chapters."""
        print("\n=== Testing Chapter Navigation ===")
        
        chapters = [
            "Chapter 0: Before Neural Networks",
            "Chapter 1: The Statistical Era",
            "Chapter 2: The Rise of Neural Networks & Embeddings",
            "Chapter 3: Sequential Models & The Power of Context",
            "Chapter 4: The Transformer Revolution",
            "Chapter 5: Applying the Foundations: Text Classification",
            "Chapter 6: The Rise of Generative Models",
            "Chapter 7: Build Your Own Generative Model",
            "Chapter 8: The Era of Large Language Models (LLMs)",
            "Chapter 9: Course Completion & Future Directions"
        ]
        
        for chapter in chapters:
            try:
                # Find and click the radio button for this chapter
                radio_selector = f"//label[contains(text(), '{chapter}')]"
                element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, radio_selector))
                )
                element.click()
                time.sleep(2)  # Wait for page to load
                
                # Verify chapter loaded by checking header
                header = self.driver.find_element(By.TAG_NAME, "h1")
                if chapter.replace(":", " -") in header.text or header.text in chapter:
                    print(f"PASS: {chapter} loaded successfully")
                    self.results["chapters"][chapter] = True
                else:
                    print(f"FAIL: {chapter} did not load properly")
                    self.results["chapters"][chapter] = False
                    
            except Exception as e:
                print(f"FAIL: Error navigating to {chapter}: {e}")
                self.results["chapters"][chapter] = False
    
    def test_coding_exercises(self):
        """Test coding practice buttons in chapters."""
        print("\n=== Testing Coding Exercises ===")
        
        chapters_with_exercises = {
            "Chapter 1: The Statistical Era": ["Build a Simple Bigram Model", "Add-1 Smoothing"],
            "Chapter 2: The Rise of Neural Networks & Embeddings": ["Calculate Cosine Similarity", "Word Analogies with Vectors"],
            "Chapter 3: Sequential Models & The Power of Context": ["Simple RNN Forward Pass"],
            "Chapter 4: The Transformer Revolution": ["Implement Scaled Dot-Product Attention", "Implement Positional Encoding"]
        }
        
        for chapter, exercises in chapters_with_exercises.items():
            try:
                # Navigate to chapter
                radio_selector = f"//label[contains(text(), '{chapter}')]"
                element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, radio_selector))
                )
                element.click()
                time.sleep(2)
                
                # Find coding practice section
                coding_section = self.driver.find_element(By.XPATH, "//h3[contains(text(), 'Coding Practice')]")
                if coding_section:
                    print(f"Found coding practice section in {chapter}")
                    
                    # Test each exercise
                    for exercise in exercises:
                        try:
                            # Find and click Start button for this exercise
                            exercise_row = self.driver.find_element(
                                By.XPATH, f"//strong[contains(text(), '{exercise}')]/ancestor::div[contains(@class, 'row-widget')]"
                            )
                            start_button = exercise_row.find_element(By.XPATH, ".//button[text()='Start']")
                            start_button.click()
                            time.sleep(2)
                            
                            # Check if exercise interface loaded
                            # Look for exercise title in h2 or fill-in blanks
                            exercise_loaded = False
                            try:
                                exercise_title = self.driver.find_element(By.XPATH, f"//h2[contains(text(), '{exercise}')]")
                                exercise_loaded = True
                            except:
                                # Check for BLANK inputs
                                blanks = self.driver.find_elements(By.XPATH, "//input[contains(@placeholder, 'BLANK')]")
                                if blanks:
                                    exercise_loaded = True
                            
                            if exercise_loaded:
                                print(f"PASS: {exercise} exercise loaded successfully")
                                self.results["coding_exercises"][f"{chapter} - {exercise}"] = True
                                
                                # Exit exercise
                                exit_button = self.driver.find_element(By.XPATH, "//button[text()='Exit Exercise']")
                                exit_button.click()
                                time.sleep(1)
                            else:
                                print(f"FAIL: {exercise} exercise did not load")
                                self.results["coding_exercises"][f"{chapter} - {exercise}"] = False
                                
                        except Exception as e:
                            print(f"FAIL: Error testing {exercise}: {e}")
                            self.results["coding_exercises"][f"{chapter} - {exercise}"] = False
                            
            except Exception as e:
                print(f"FAIL: Error testing coding exercises in {chapter}: {e}")
    
    def test_quiz_interface(self):
        """Test the quiz interface."""
        print("\n=== Testing Quiz Interface ===")
        
        try:
            # Navigate to quizzes
            radio_selector = "//label[contains(text(), 'Knowledge Check Quizzes')]"
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, radio_selector))
            )
            element.click()
            time.sleep(2)
            
            # Check if quiz interface loaded
            quiz_header = self.driver.find_element(By.XPATH, "//h2[contains(text(), 'Interactive Code Exercises') or contains(text(), 'Knowledge Check')]")
            if quiz_header:
                print("PASS: Quiz interface loaded")
                self.results["quizzes"]["interface_loaded"] = True
                
                # Try to start a quiz
                start_buttons = self.driver.find_elements(By.XPATH, "//button[text()='Start Quiz']")
                if start_buttons:
                    start_buttons[0].click()
                    time.sleep(2)
                    
                    # Check if quiz questions appeared
                    questions = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'stRadio')]")
                    if questions:
                        print(f"PASS: Quiz started with {len(questions)} questions")
                        self.results["quizzes"]["quiz_started"] = True
                    else:
                        print("FAIL: No quiz questions found")
                        self.results["quizzes"]["quiz_started"] = False
                else:
                    print("INFO: No quiz start buttons found")
                    self.results["quizzes"]["quiz_started"] = "No quizzes available"
            else:
                print("FAIL: Quiz interface did not load")
                self.results["quizzes"]["interface_loaded"] = False
                
        except Exception as e:
            print(f"FAIL: Error testing quiz interface: {e}")
            self.results["quizzes"]["interface_loaded"] = False
    
    def test_glossary(self):
        """Test the glossary functionality."""
        print("\n=== Testing Glossary ===")
        
        try:
            # Navigate to glossary
            radio_selector = "//label[contains(text(), 'NLP Glossary')]"
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, radio_selector))
            )
            element.click()
            time.sleep(2)
            
            # Check if glossary loaded
            glossary_header = self.driver.find_element(By.XPATH, "//h1[contains(text(), 'NLP Glossary')]")
            if glossary_header:
                print("PASS: Glossary loaded")
                
                # Test search functionality
                search_input = self.driver.find_element(By.XPATH, "//input[contains(@placeholder, 'Type to search')]")
                if search_input:
                    search_input.send_keys("transformer")
                    time.sleep(1)
                    
                    # Check if results filtered
                    expanders = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'streamlit-expanderHeader')]")
                    if expanders:
                        print(f"PASS: Glossary search working, found {len(expanders)} results")
                        self.results["glossary"] = True
                    else:
                        print("FAIL: No glossary search results")
                        self.results["glossary"] = False
                else:
                    print("FAIL: Glossary search input not found")
                    self.results["glossary"] = False
            else:
                print("FAIL: Glossary did not load")
                self.results["glossary"] = False
                
        except Exception as e:
            print(f"FAIL: Error testing glossary: {e}")
            self.results["glossary"] = False
    
    def test_dataset_explorer(self):
        """Test the dataset explorer."""
        print("\n=== Testing Dataset Explorer ===")
        
        try:
            # Navigate to dataset explorer
            radio_selector = "//label[contains(text(), 'Dataset Explorer')]"
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, radio_selector))
            )
            element.click()
            time.sleep(2)
            
            # Check if dataset explorer loaded
            explorer_header = self.driver.find_element(By.XPATH, "//h2[contains(text(), 'Dataset Explorer')]")
            if explorer_header:
                print("PASS: Dataset Explorer loaded")
                
                # Check for dataset options
                selectboxes = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'stSelectbox')]")
                if selectboxes:
                    print(f"PASS: Dataset Explorer has {len(selectboxes)} selection options")
                    self.results["dataset_explorer"] = True
                else:
                    print("INFO: Dataset Explorer loaded but no datasets available")
                    self.results["dataset_explorer"] = True
            else:
                print("FAIL: Dataset Explorer did not load")
                self.results["dataset_explorer"] = False
                
        except Exception as e:
            print(f"FAIL: Error testing dataset explorer: {e}")
            self.results["dataset_explorer"] = False
    
    def generate_report(self):
        """Generate a test report."""
        print("\n=== TEST REPORT ===")
        
        # Chapter navigation results
        chapter_passes = sum(1 for v in self.results["chapters"].values() if v)
        print(f"\nChapter Navigation: {chapter_passes}/{len(self.results['chapters'])} passed")
        
        # Coding exercises results
        exercise_passes = sum(1 for v in self.results["coding_exercises"].values() if v)
        print(f"Coding Exercises: {exercise_passes}/{len(self.results['coding_exercises'])} passed")
        
        # Other components
        print(f"Quiz Interface: {'PASS' if self.results['quizzes'].get('interface_loaded') else 'FAIL'}")
        print(f"Glossary: {'PASS' if self.results['glossary'] else 'FAIL'}")
        print(f"Dataset Explorer: {'PASS' if self.results['dataset_explorer'] else 'FAIL'}")
        
        # Overall result
        total_tests = (
            len(self.results["chapters"]) + 
            len(self.results["coding_exercises"]) + 
            3  # quiz, glossary, dataset explorer
        )
        total_passes = (
            chapter_passes + 
            exercise_passes + 
            (1 if self.results['quizzes'].get('interface_loaded') else 0) +
            (1 if self.results['glossary'] else 0) +
            (1 if self.results['dataset_explorer'] else 0)
        )
        
        print(f"\nOVERALL: {total_passes}/{total_tests} tests passed")
        
        return total_passes == total_tests
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
        
        if self.app_process:
            if sys.platform == 'win32':
                self.app_process.terminate()
            else:
                os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
            self.app_process.wait()
    
    def run(self):
        """Run the complete end-to-end test."""
        try:
            # Start app
            if not self.start_app():
                print("Failed to start app")
                return False
            
            # Setup WebDriver
            if not self.setup_driver():
                print("Failed to setup WebDriver")
                return False
            
            # Run tests
            self.test_chapter_navigation()
            self.test_coding_exercises()
            self.test_quiz_interface()
            self.test_glossary()
            self.test_dataset_explorer()
            
            # Generate report
            success = self.generate_report()
            
            return success
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            return False
        finally:
            self.cleanup()


def run_simplified_test():
    """Run a simplified test without Selenium (checks basic functionality)."""
    print("=== Running Simplified E2E Test (No Selenium) ===\n")
    
    results = {
        "imports": [],
        "chapters": [],
        "exercises": [],
        "components": []
    }
    
    # Test 1: Can import all modules
    print("Testing module imports...")
    modules = [
        "nlp_evolution_app",
        "chapter0", "chapter1", "chapter2", "chapter3", "chapter4",
        "chapter5", "chapter6", "chapter7", "chapter8", "chapter9",
        "quiz_system", "code_exercises", "datasets", "glossary",
        "theme_manager", "utils", "sklearn_fallbacks"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"PASS: {module} imported successfully")
            results["imports"].append(True)
        except Exception as e:
            print(f"FAIL: {module} import failed: {e}")
            results["imports"].append(False)
    
    # Test 2: Check chapter render functions
    print("\nTesting chapter render functions...")
    for i in range(10):
        try:
            if i == 0:
                from chapter0 import render_chapter_0
                render_func = render_chapter_0
            else:
                module = __import__(f"chapter{i}")
                render_func = getattr(module, f"render_chapter_{i}")
            
            print(f"PASS: Chapter {i} render function exists")
            results["chapters"].append(True)
        except Exception as e:
            print(f"FAIL: Chapter {i} render function error: {e}")
            results["chapters"].append(False)
    
    # Test 3: Check coding exercises
    print("\nTesting coding exercises...")
    try:
        from code_exercises import CODE_EXERCISES, CodeExerciseManager
        
        manager = CodeExerciseManager()
        for chapter, data in CODE_EXERCISES.items():
            exercise_count = len(data.get("exercises", []))
            if exercise_count > 0:
                print(f"PASS: {chapter} has {exercise_count} exercises")
                results["exercises"].append(True)
                
                # Test starting first exercise
                first_exercise = data["exercises"][0]
                if manager.start_exercise(chapter, first_exercise["id"]):
                    print(f"  - Can start exercise: {first_exercise['title']}")
            else:
                print(f"INFO: {chapter} has no exercises")
    except Exception as e:
        print(f"FAIL: Error testing exercises: {e}")
        results["exercises"].append(False)
    
    # Test 4: Check component interfaces
    print("\nTesting component interfaces...")
    components = [
        ("quiz_system", "render_quiz_interface"),
        ("code_exercises", "render_code_exercise_interface"),
        ("datasets", "render_dataset_explorer"),
        ("glossary", "render_glossary")
    ]
    
    for module_name, func_name in components:
        try:
            module = __import__(module_name)
            func = getattr(module, func_name)
            print(f"PASS: {func_name} exists in {module_name}")
            results["components"].append(True)
        except Exception as e:
            print(f"FAIL: {func_name} in {module_name}: {e}")
            results["components"].append(False)
    
    # Generate report
    print("\n=== SIMPLIFIED TEST REPORT ===")
    
    import_passes = sum(results["imports"])
    chapter_passes = sum(results["chapters"])
    exercise_passes = sum(results["exercises"])
    component_passes = sum(results["components"])
    
    print(f"Module Imports: {import_passes}/{len(results['imports'])} passed")
    print(f"Chapter Functions: {chapter_passes}/{len(results['chapters'])} passed")
    print(f"Coding Exercises: {exercise_passes}/{len(results['exercises'])} passed")
    print(f"Component Interfaces: {component_passes}/{len(results['components'])} passed")
    
    total_tests = sum(len(r) for r in results.values())
    total_passes = import_passes + chapter_passes + exercise_passes + component_passes
    
    print(f"\nOVERALL: {total_passes}/{total_tests} tests passed")
    
    return total_passes == total_tests


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-end test for NLP Evolution app")
    parser.add_argument("--full", action="store_true", help="Run full test with Selenium (requires Chrome)")
    parser.add_argument("--show-browser", action="store_true", help="Show browser window during test")
    
    args = parser.parse_args()
    
    if args.full:
        print("Running full E2E test with Selenium...")
        print("This requires Chrome and ChromeDriver to be installed.")
        print("Install ChromeDriver: https://chromedriver.chromium.org/\n")
        
        tester = NLPEvolutionE2ETest(headless=not args.show_browser)
        success = tester.run()
    else:
        print("Running simplified test (no browser automation)...")
        print("For full browser testing, use: python test_e2e.py --full\n")
        
        success = run_simplified_test()
    
    sys.exit(0 if success else 1)