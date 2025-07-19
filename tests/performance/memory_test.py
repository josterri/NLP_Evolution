from memory_profiler import profile
import sys
sys.path.append(".")
from nlp_evolution_app import *

@profile
def test_memory_usage():
    """Test memory usage of key functions"""
    # Add your memory-intensive operations here
    # Example:
    # process_large_text()
    # analyze_sentiment()
    pass

if __name__ == "__main__":
    test_memory_usage() 