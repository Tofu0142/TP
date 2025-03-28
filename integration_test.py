import unittest
import requests
import time
import subprocess
import os
import signal

class TestModelServerIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server_process = subprocess.Popen(['python', 'model_server.py'])
        # Wait for server to start
        time.sleep(5)
    
    @classmethod
    def tearDownClass(cls):
        # Terminate the server
        cls.server_process.terminate()
        cls.server_process.wait()
    
    def test_prediction_endpoint(self):
        # Test the prediction endpoint
        response = requests.post(
            'http://localhost:8080/predict',
            json={'text': 'This product is amazing and I love it!'}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertIn('sentiment', data)
        self.assertIn('confidence', data)
        self.assertIn('latency_ms', data)
        
        # Check prediction is as expected
        self.assertEqual(data['sentiment'], 'positive')
        self.assertGreater(data['confidence'], 0.5)
    
    def test_metrics_endpoint(self):
        # Test the metrics endpoint
        response = requests.get('http://localhost:8080/metrics')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check metrics structure
        self.assertIn('p99_latency_ms', data)
        self.assertIn('avg_latency_ms', data)
        self.assertIn('requests_per_second', data)
        self.assertIn('total_requests', data)

if __name__ == '__main__':
    unittest.main() 