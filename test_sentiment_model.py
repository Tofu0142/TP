import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from model_server import preprocess_input, predict, PerformanceMonitor
from review_prediction import BertSentimentClassifier

class TestSentimentModel(unittest.TestCase):
    
    def setUp(self):
        # Setup test environment
        self.model = MagicMock()
        self.device = torch.device('cpu')
        
    def test_preprocess_input(self):
        # Test preprocessing function
        with patch('model_server.tokenizer') as mock_tokenizer, \
             patch('model_server.device', self.device):
            
            mock_tokenizer.return_value = {
                'input_ids': torch.ones((1, 128)),
                'attention_mask': torch.ones((1, 128))
            }
            
            input_ids, attention_mask, numerical_features = preprocess_input("Test sentence")
            
            self.assertEqual(input_ids.shape, torch.Size([1, 128]))
            self.assertEqual(attention_mask.shape, torch.Size([1, 128]))
            self.assertEqual(numerical_features.shape, torch.Size([1, 5]))
    
    def test_predict(self):
        # Test prediction function
        with patch('model_server.model') as mock_model:
            mock_outputs = torch.tensor([[0.1, 0.2, 0.7]])
            mock_model.return_value = mock_outputs
            
            input_ids = torch.ones((1, 128))
            attention_mask = torch.ones((1, 128))
            numerical_features = torch.ones((1, 5))
            
            prediction, confidence = predict(input_ids, attention_mask, numerical_features)
            
            self.assertEqual(prediction, 'positive')
            self.assertAlmostEqual(confidence, 0.7, places=1)
    
    def test_performance_monitor(self):
        # Test performance monitoring
        monitor = PerformanceMonitor(window_size=10)
        
        # Add some test latencies
        test_latencies = [10, 20, 30, 40, 50]
        for latency in test_latencies:
            monitor.add_latency(latency)
        
        # Test metrics calculations
        self.assertEqual(monitor.get_avg_latency(), 30.0)
        self.assertEqual(monitor.get_p99_latency(), 49.6)

if __name__ == '__main__':
    unittest.main() 