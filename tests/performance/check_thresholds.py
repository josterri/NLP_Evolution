import json
import sys
from pathlib import Path

def load_thresholds():
    """Load threshold values from configuration"""
    threshold_file = Path(__file__).parent / "thresholds.json"
    with open(threshold_file) as f:
        return json.load(f)

def check_memory_profile(profile_file, thresholds):
    """Check memory profile against thresholds"""
    with open(profile_file) as f:
        lines = f.readlines()
    
    # Parse memory usage from profile output
    max_mem = 0
    for line in lines:
        if "MiB" in line:
            try:
                mem = float(line.split()[3])
                max_mem = max(max_mem, mem)
            except (IndexError, ValueError):
                continue
    
    if max_mem > thresholds["memory"]["max_usage_mb"]:
        print(f"❌ Memory usage ({max_mem}MB) exceeds threshold ({thresholds['memory']['max_usage_mb']}MB)")
        return False
    
    print(f"✅ Memory usage ({max_mem}MB) within threshold")
    return True

def check_load_test(results_file, thresholds):
    """Check load test results against thresholds"""
    with open(results_file) as f:
        lines = f.readlines()
    
    # Parse Locust results
    for line in lines:
        if "Requests/second" in line:
            rps = float(line.split()[-1])
            if rps < thresholds["load_test"]["min_requests_per_second"]:
                print(f"❌ Requests per second ({rps}) below threshold ({thresholds['load_test']['min_requests_per_second']})")
                return False
    
    print("✅ Load test metrics within thresholds")
    return True

def check_benchmark(benchmark_file, thresholds):
    """Check benchmark results against thresholds"""
    with open(benchmark_file) as f:
        data = json.load(f)
    
    mean_time = data["benchmarks"][0]["stats"]["mean"]
    std_dev = data["benchmarks"][0]["stats"]["stddev"]
    
    if mean_time > thresholds["benchmark"]["max_mean_time_ms"]:
        print(f"❌ Mean time ({mean_time}ms) exceeds threshold ({thresholds['benchmark']['max_mean_time_ms']}ms)")
        return False
    
    if std_dev > thresholds["benchmark"]["max_std_dev_ms"]:
        print(f"❌ Standard deviation ({std_dev}ms) exceeds threshold ({thresholds['benchmark']['max_std_dev_ms']}ms)")
        return False
    
    print("✅ Benchmark metrics within thresholds")
    return True

def main():
    """Main function to check all performance metrics"""
    thresholds = load_thresholds()
    
    memory_ok = check_memory_profile("memory_profile.txt", thresholds)
    load_ok = check_load_test("locust_results.txt", thresholds)
    benchmark_ok = check_benchmark("output.json", thresholds)
    
    if not all([memory_ok, load_ok, benchmark_ok]):
        sys.exit(1)

if __name__ == "__main__":
    main() 