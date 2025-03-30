import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from model_server import preprocess_input, predict_sentiment_internal, PerformanceMonitor
import pandas as pd

class TestSentimentModel(unittest.TestCase):
    
    def setUp(self):
        # Setup test environment
        self.model = MagicMock()
        
    def test_preprocess_input(self):
        # Test preprocessing function
        with patch('model_server.tfidf_vectorizer') as mock_tfidf, \
             patch('model_server.doc2vec_model') as mock_doc2vec:
            
            # Mock TF-IDF vectorizer
            mock_tfidf.transform.return_value = np.zeros((1, 1000))
            
            # Mock Doc2Vec model
            mock_doc2vec.infer_vector.return_value = np.zeros(100)
            
            # Call the function
            with patch('model_server.TextBlob') as mock_textblob:
                mock_sentiment = MagicMock()
                mock_sentiment.polarity = 0.5
                mock_textblob.return_value.sentiment = mock_sentiment
                
                features = preprocess_input("Test sentence")
                
                # Check that the result is a DataFrame with expected columns
                self.assertIsInstance(features, pd.DataFrame)
                self.assertEqual(features.shape[0], 1)  # One row
                self.assertIn('body_sentiment', features.columns)
                self.assertIn('title_sentiment', features.columns)
    
    def test_predict_sentiment_internal(self):
        # Test prediction function
        with patch('model_server.preprocess_input') as mock_preprocess, \
             patch('model_server.model') as mock_model:
            
            # Mock preprocessing
            mock_preprocess.return_value = pd.DataFrame({'feature': [1.0]})
            
            # Mock model prediction
            mock_model.predict.return_value = [1]  # neutral
            mock_model.predict_proba.return_value = [[0.1, 0.7, 0.2]]
            
            # Call the function
            prediction, confidence = predict_sentiment_internal("Test sentence")
            
            # Check results
            self.assertEqual(prediction, 'neutral')
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