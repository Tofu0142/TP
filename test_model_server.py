import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import sys
import os
import subprocess
import signal
import pytest
import socket
from contextlib import closing

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

def find_free_port():
    """Find a free port to use for the server."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

@pytest.fixture(scope="module")
def server():
    """Start the server for testing and shut it down after."""
    # Find a free port
    port = find_free_port()
    
    # Start the server as a subprocess
    server_process = subprocess.Popen(
        [sys.executable, "model_server.py"],
        env=dict(os.environ, PORT=str(port)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)  # Give the server some time to start
    
    # Check if server is running
    try:
        response = requests.get(f"http://localhost:{port}/health")
        if response.status_code != 200:
            pytest.fail(f"Server failed to start properly: {response.text}")
    except requests.exceptions.ConnectionError:
        # If we can't connect, kill the server and fail the test
        server_process.terminate()
        stdout, stderr = server_process.communicate()
        pytest.fail(f"Failed to connect to server: {stderr.decode()}")
    
    # Return the server process and port
    yield {"process": server_process, "port": port}
    
    # Shutdown the server
    server_process.terminate()
    server_process.wait()

def test_model_server(server):
    """Test the model server with real requests and measure latency."""
    port = server["port"]
    url = f"http://localhost:{port}/predict"
    num_requests = 100
    latencies = []
    
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
    plt.savefig("test_latency_distribution.png")
    
    # Assert that p99 latency is below 300ms
    assert p99_latency < 300, f"P99 latency ({p99_latency:.2f}ms) exceeds target (300ms)"
    
    # Test other endpoints
    # Test /metrics endpoint
    response = requests.get(f"http://localhost:{port}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert 'total_requests' in metrics
    assert 'avg_latency_ms' in metrics
    
    # Test /health endpoint
    response = requests.get(f"http://localhost:{port}/health")
    assert response.status_code == 200
    health = response.json()
    assert health['status'] == 'healthy'
    assert 'uptime' in health
    
    # Test batch prediction
    response = requests.post(
        f"http://localhost:{port}/predict_batch",
        json={"texts": test_reviews[:3]}
    )
    assert response.status_code == 200
    batch_result = response.json()
    assert len(batch_result['results']) == 3
    assert batch_result['count'] == 3
    assert 'latency_ms' in batch_result

if __name__ == "__main__":
    # This allows running the test directly
    server_fixture = next(server())
    try:
        test_model_server(server_fixture)
    finally:
        server_fixture["process"].terminate()
        server_fixture["process"].wait() 