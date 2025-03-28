import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import argparse
from tqdm import tqdm

def send_request(text):
    """Send a single request to the model server"""
    start_time = time.time()
    try:
        response = requests.post(
            'http://localhost:8080/predict',
            json={'text': text},
            timeout=10
        )
        latency = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            return {
                'success': True,
                'latency': latency,
                'sentiment': response.json()['sentiment']
            }
        else:
            return {
                'success': False,
                'latency': latency,
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        latency = (time.time() - start_time) * 1000  # ms
        return {
            'success': False,
            'latency': latency,
            'error': str(e)
        }

def load_test(concurrency=10, requests_per_worker=10, ramp_up=5):
    """
    Run a load test with specified concurrency
    
    Args:
        concurrency: Number of concurrent workers
        requests_per_worker: Number of requests each worker sends
        ramp_up: Seconds to ramp up to full concurrency
    """
    # Sample texts of varying sentiment
    texts = [
        "This product is amazing! I love everything about it.",
        "It's okay, nothing special but gets the job done.",
        "Terrible experience. Would not recommend to anyone.",
        "Decent quality for the price, but could be better.",
        "Absolutely fantastic service and product quality!"
    ]
    
    total_requests = concurrency * requests_per_worker
    results = []
    
    print(f"Starting load test with {concurrency} concurrent workers, {total_requests} total requests")
    
    # Create a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        
        # Submit tasks with ramp-up
        for i in range(concurrency):
            # Calculate delay for ramp-up
            if ramp_up > 0:
                delay = (i / concurrency) * ramp_up
                time.sleep(delay / concurrency)
            
            # Each worker sends multiple requests
            for j in range(requests_per_worker):
                text = texts[np.random.randint(0, len(texts))]
                futures.append(executor.submit(send_request, text))
        
        # Collect results with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    
    # Analyze results
    latencies = [r['latency'] for r in results]
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    
    print(f"\nResults:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Latency: {np.mean(latencies):.2f} ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
    
    # Plot latency distribution
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, alpha=0.7)
    plt.axvline(x=np.percentile(latencies, 95), color='r', linestyle='--', 
                label=f'P95: {np.percentile(latencies, 95):.2f} ms')
    plt.axvline(x=np.percentile(latencies, 99), color='g', linestyle='--', 
                label=f'P99: {np.percentile(latencies, 99):.2f} ms')
    plt.axvline(x=np.mean(latencies), color='b', linestyle='--', 
                label=f'Mean: {np.mean(latencies):.2f} ms')
    plt.title('Load Test Latency Distribution')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("load_test_results.png")
    plt.close()
    
    return {
        
        'avg_latency': np.mean(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load test the sentiment analysis model server')
    parser.add_argument('--concurrency', type=int, default=10, help='Number of concurrent workers')
    parser.add_argument('--requests', type=int, default=10, help='Requests per worker')
    parser.add_argument('--ramp-up', type=float, default=5, help='Ramp-up time in seconds')
    
    args = parser.parse_args()
    load_test(args.concurrency, args.requests, args.ramp_up) 