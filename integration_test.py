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
            stderr=subprocess.PIPE
        )
        
        # Give the server a moment to start up
        time.sleep(3)
        
        # Check if server started successfully
        if cls.server_process.poll() is not None:
            # Server failed to start
            stdout, stderr = cls.server_process.communicate()
            print(f"Server failed to start: {stderr.decode('utf-8')}")
            sys.exit(1)
        
        print("Server started successfully")
        
        # Try to connect to the server to verify it's running
        max_retries = 5
        for i in range(max_retries):
            try:
                requests.get('http://localhost:8080/', timeout=1)
                print("Successfully connected to server")
                break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    print(f"Server not ready yet, retrying in 1 second... ({i+1}/{max_retries})")
                    time.sleep(1)
                else:
                    print("Failed to connect to server after multiple attempts")
                    cls.tearDownClass()
                    sys.exit(1)
    
    def test_predict_endpoint(self):
        # Test the prediction endpoint with positive sentiment
        payload = {"text": "I love this product, it's amazing!"}
        response = requests.post('http://localhost:8080/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('sentiment', data)
        self.assertIn('confidence', data)
        self.assertIn('processing_time', data)
    
    def test_negative_sentiment(self):
        # Test with negative sentiment
        payload = {"text": "This is terrible, I hate it."}
        response = requests.post('http://localhost:8080/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('sentiment', data)
    
    def test_metrics_endpoint(self):
        # Test the metrics endpoint
        response = requests.get('http://localhost:8080/metrics')
        self.assertEqual(response.status_code, 200)
        self.assertIn('sentiment_predictions_total', response.text)
    
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
            if stderr:
                print(f"Server stderr: {stderr.decode('utf-8')}")
            
            print("Server shutdown complete")

if __name__ == '__main__':
    unittest.main() 