import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Test data
test_reviews = [
    "This book was absolutely fantastic! I couldn't put it down.",
    "The characters were poorly developed and the plot was predictable.",
    "It was an okay read, nothing special but not terrible either.",
    "I loved the author's writing style but the ending was disappointing.",
    "The book started strong but lost its way in the middle chapters.",
    "One of the best books I've read this year, highly recommend!",
    "The premise was interesting but the execution was terrible.",
    "A solid read with well-developed characters and an engaging plot.",
    "I found the book to be quite boring and struggled to finish it.",
    "The world-building was incredible, but the pacing was too slow."
]

def test_model_server(num_requests=100):
    url = "http://localhost:8080/predict"
    latencies = []
    results = []
    
    print(f"Sending {num_requests} requests to the model server...")
    
    for _ in tqdm(range(num_requests)):
        # Select a random review
        review_idx = np.random.randint(0, len(test_reviews))
        review = test_reviews[review_idx]
        
        # Prepare request
        payload = {"text": review}
        
        # Send request and measure time
        start_time = time.time()
        response = requests.post(url, json=payload)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Store results
        if response.status_code == 200:
            result = response.json()
            results.append(result)
            latencies.append(latency)
        else:
            print(f"Error: {response.status_code}, {response.text}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print("\nResults:")
    print(f"Total requests: {len(latencies)}")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")
    print(f"P99 latency: {p99_latency:.2f}ms")
    
    # Plot latency distribution
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, alpha=0.7)
    plt.axvline(x=p99_latency, color='r', linestyle='--', label=f'P99: {p99_latency:.2f}ms')
    plt.axvline(x=p95_latency, color='g', linestyle='--', label=f'P95: {p95_latency:.2f}ms')
    plt.axvline(x=avg_latency, color='b', linestyle='--', label=f'Avg: {avg_latency:.2f}ms')
    plt.axvline(x=300, color='k', linestyle='-', label='Target: 300ms')
    plt.title('Request Latency Distribution')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("client_latency_distribution.png")
    plt.show()
    
    # Return results
    return {
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "p99_latency": p99_latency,
        "results": results
    }

if __name__ == "__main__":
    test_model_server(100) 