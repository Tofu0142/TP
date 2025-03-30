import unittest
import requests
import time
import threading
import sys
import os
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_server

class TestModelServerIntegration(unittest.TestCase):
    """Integration tests for the model server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class - start the server in a separate thread."""
        
        cls.server_thread = None
        cls.app = model_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
    
    def test_predict_endpoint(self):
        """Test the /predict endpoint."""
        with patch('model_server.predict_sentiment_internal') as mock_predict:
           
            mock_predict.return_value = ('positive', 0.9)
            
            
            response = self.client.post('/predict', 
                                      json={'text': 'This is a test.'})
            
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['sentiment'], 'positive')
            self.assertEqual(data['confidence'], 0.9)
            self.assertIn('latency_ms', data)
    
    def test_predict_batch_endpoint(self):
        """Test the /predict_batch endpoint."""
        with patch('model_server.predict_sentiment_internal') as mock_predict:
            
            mock_predict.return_value = ('positive', 0.9)
            
            
            response = self.client.post('/predict_batch', 
                                      json={'texts': ['Test 1', 'Test 2']})
            
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(len(data['results']), 2)
            self.assertEqual(data['count'], 2)
            self.assertIn('latency_ms', data)
    
    def test_metrics_endpoint(self):
        """Test the /metrics endpoint."""
       
        response = self.client.get('/metrics')
        
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('total_requests', data)
        self.assertIn('avg_latency_ms', data)
    
    def test_health_endpoint(self):
        """Test the /health endpoint."""
        
        response = self.client.get('/health')
        
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('uptime', data)

if __name__ == '__main__':
    unittest.main() 