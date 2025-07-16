import pytest
import sys
sys.path.append(".")
from nlp_evolution_app import *

def test_performance_benchmark(benchmark):
    """Benchmark key operations"""
    def run_benchmark():
        # Add your performance-critical operations here
        # Example:
        # result = process_text("sample text")
        pass
    
    # Run the benchmark
    benchmark(run_benchmark) 