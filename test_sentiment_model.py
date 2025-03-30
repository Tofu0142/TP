import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

class TestSentimentModel(unittest.TestCase):
    
    def setUp(self):
        # Setup test environment
        self.device = torch.device('cpu')
    
    @patch('model_server.tokenizer')
    def test_preprocess_input(self, mock_tokenizer):
        # Configure the mock tokenizer
        mock_encoded = {
            'input_ids': torch.ones((1, 128)),
            'attention_mask': torch.ones((1, 128))
        }
        mock_tokenizer.return_value = mock_encoded
        
        # Call the function with a test input
        with patch('model_server.device', self.device):
            from model_server import preprocess_input
            input_ids, attention_mask, numerical_features = preprocess_input("Test sentence")
        
        # Verify the results
        self.assertEqual(input_ids.shape, torch.Size([1, 128]))
        self.assertEqual(attention_mask.shape, torch.Size([1, 128]))
    
    @patch('model_server.using_fallback_model', True)
    @patch('model_server.fallback_vectorizer')
    @patch('model_server.fallback_model')
    def test_fallback_prediction(self, mock_model, mock_vectorizer):
        # Configure the mocks
        mock_transform = MagicMock()
        mock_vectorizer.transform.return_value = mock_transform
        mock_model.predict.return_value = [2]  # 2 = positive
        mock_model.predict_proba.return_value = [[0.1, 0.2, 0.7]]
        
        # Import the function
        from model_server import predict_sentiment
        
        # Mock the Flask request context
        with patch('model_server.request') as mock_request:
            mock_request.json = {"text": "This is a test"}
            
            # Call the function
            result = predict_sentiment()
            
            # Check the result
            self.assertIn('sentiment', result)
    
    def test_performance_monitor(self):
        # Import the class
        from model_server import PerformanceMonitor
        
        # Create an instance
        monitor = PerformanceMonitor(window_size=10)
        
        # Add some test latencies
        test_latencies = [10, 20, 30, 40, 50]
        for latency in test_latencies:
            monitor.add_latency(latency)
        
        # Test metrics calculations
        self.assertEqual(monitor.get_avg_latency(), 30.0)
        self.assertIn('p99_latency_ms', monitor.generate_report())

if __name__ == '__main__':
    unittest.main() 