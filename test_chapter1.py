#!/usr/bin/env python3
"""
Test Chapter 1 functionality specifically to check for st variable errors.
"""

import subprocess
import time
import sys
import requests
import threading

def test_chapter1_loading():
    """Test if Chapter 1 can load without st variable errors."""
    print("Testing Chapter 1 functionality...")
    
    try:
        # Start the app process with a specific port
        port = 8512
        proc = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'nlp_evolution_app.py', '--server.port', str(port)],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup
        time.sleep(8)
        
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"FAIL: App failed to start")
            print(f"Return code: {proc.returncode}")
            print(f"Stderr: {stderr}")
            return False
        
        # Try to access the app
        try:
            response = requests.get(f"http://localhost:{port}", timeout=5)
            if response.status_code == 200:
                print("PASS: App is running and accessible")
                success = True
            else:
                print(f"WARN: App running but returned status {response.status_code}")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"WARN: Could not connect to app: {e}")
            success = False
        
        # Check for errors in stderr
        def check_stderr():
            try:
                for line in proc.stderr:
                    if "cannot access local variable 'st'" in line:
                        print(f"FAIL: Found st variable error: {line.strip()}")
                        return False
                    elif "ERROR" in line or "Exception" in line:
                        print(f"WARN: Found error in stderr: {line.strip()}")
            except:
                pass
        
        # Start stderr monitoring in background
        stderr_thread = threading.Thread(target=check_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()
        
        # Let it run for a few seconds to catch any errors
        time.sleep(3)
        
        # Clean shutdown
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        
        return success
        
    except Exception as e:
        print(f"FAIL: Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_chapter1_loading()
    print(f"Chapter 1 test: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)