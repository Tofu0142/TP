import unittest
import requests
import time
import subprocess
import os
import signal
import sys

class TestModelServerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the server as a subprocess
        print("Starting server for integration tests...")
        cls.server_process = subprocess.Popen(
            ['python', 'model_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Use text mode for easier output handling
            bufsize=1   # Line buffered
        )
        
        # Give the server a moment to start up
        time.sleep(5)  # Increased wait time
        
        # Check if server started successfully
        if cls.server_process.poll() is not None:
            # Server failed to start
            stdout, stderr = cls.server_process.communicate()
            print(f"Server failed to start!")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            raise RuntimeError("Server failed to start")
        
        print("Server process started")
        
        # Try to connect to the server to verify it's running
        max_retries = 10  # Increased retries
        for i in range(max_retries):
            try:
                response = requests.get('http://localhost:8080/', timeout=2)
                print(f"Successfully connected to server: {response.status_code}")
                break
            except requests.exceptions.ConnectionError as e:
                if i < max_retries - 1:
                    print(f"Server not ready yet, retrying in 2 seconds... ({i+1}/{max_retries})")
                    print(f"Connection error: {str(e)}")
                    
                    # Check if process is still running
                    if cls.server_process.poll() is not None:
                        stdout, stderr = cls.server_process.communicate()
                        print(f"Server process terminated unexpectedly!")
                        print(f"STDOUT: {stdout}")
                        print(f"STDERR: {stderr}")
                        raise RuntimeError("Server process terminated unexpectedly")
                    
                    time.sleep(2)
                else:
                    print("Failed to connect to server after multiple attempts")
                    # Instead of exiting, raise an exception for better test reporting
                    cls.tearDownClass()
                    raise RuntimeError("Failed to connect to server after multiple attempts")
    
    def test_predict_endpoint(self):
        # Test the predict endpoint with a positive sentiment
        payload = {"text": "This is great!"}
        response = requests.post('http://localhost:8080/predict', json=payload)
        
        # Print response for debugging
        print(f"Predict endpoint response: {response.status_code}")
        if response.status_code != 200:
            print(f"Response text: {response.text}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('sentiment', data)
        self.assertIn('latency_ms', data)  # Check for latency_ms instead of processing_time
        self.assertIn('confidence', data)
    
    def test_negative_sentiment(self):
        # Test with negative sentiment
        payload = {"text": "This is terrible, I hate it."}
        response = requests.post('http://localhost:8080/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('sentiment', data)
    
    def test_metrics_endpoint(self):
        # Test the Prometheus metrics endpoint
        response = requests.get('http://localhost:8080/prometheus')
        self.assertEqual(response.status_code, 200)
        
        # Check for the metrics that are actually present
        self.assertIn('sentiment_requests_total', response.text)
        self.assertIn('sentiment_request_latency_milliseconds', response.text)
        self.assertIn('sentiment_errors_total', response.text)
        self.assertIn('sentiment_model_confidence', response.text)
        self.assertIn('sentiment_results_total', response.text)
    
    @classmethod
    def tearDownClass(cls):
        # Terminate the server process
        if hasattr(cls, 'server_process') and cls.server_process:
            print("Shutting down test server...")
            
            # Try graceful termination first
            if sys.platform == 'win32':
                cls.server_process.terminate()
            else:
                os.kill(cls.server_process.pid, signal.SIGTERM)
            
            # Give it some time to shut down
            timeout = 5
            for _ in range(timeout):
                if cls.server_process.poll() is not None:
                    break
                time.sleep(1)
            
            # Force kill if still running
            if cls.server_process.poll() is None:
                print("Server didn't terminate gracefully, forcing...")
                if sys.platform == 'win32':
                    cls.server_process.kill()
                else:
                    os.kill(cls.server_process.pid, signal.SIGKILL)
            
            # Collect any output
            stdout, stderr = cls.server_process.communicate()
            if stdout:
                print(f"Server stdout: {stdout}")
            if stderr:
                print(f"Server stderr: {stderr}")
            
            print("Server shutdown complete")

if __name__ == '__main__':
    unittest.main() 