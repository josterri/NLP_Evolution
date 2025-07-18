#!/usr/bin/env python3
"""
Quick startup test for the NLP Evolution app.
"""

import subprocess
import time
import sys

def test_app_startup():
    """Test if the app can start without immediate errors."""
    print("Testing app startup...")
    
    try:
        # Start the app process
        proc = subprocess.Popen(
            [sys.executable, 'launch_app.py'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds for startup
        time.sleep(5)
        
        if proc.poll() is None:
            print("PASS: App started successfully (process running)")
            proc.terminate()
            proc.wait(timeout=5)
            return True
        else:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"FAIL: App failed to start:")
                print(f"Return code: {proc.returncode}")
                print(f"Stderr: {stderr}")
                return False
            else:
                print("WARN: App started and exited immediately")
                return False
                
    except Exception as e:
        print(f"FAIL: Error during startup test: {e}")
        return False

if __name__ == "__main__":
    success = test_app_startup()
    print(f"Startup test: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)