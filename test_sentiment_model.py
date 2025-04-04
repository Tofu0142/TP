import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from model_server import preprocess_input, predict_sentiment_internal, app
import pandas as pd
import time
from scipy.sparse import csr_matrix

class TestSentimentModel(unittest.TestCase):
    
    def setUp(self):
        # Setup test environment
        self.model = MagicMock()
        
    def test_preprocess_input(self):
        # Test preprocessing function
        with patch('model_server.tfidf_vectorizer') as mock_tfidf, \
             patch('model_server.clean_text') as mock_clean:
            
            # Mock clean_text function
            mock_clean.return_value = "cleaned text"
            
            # Mock TF-IDF vectorizer to return a sparse matrix
            mock_sparse_matrix = csr_matrix(np.zeros((1, 1000)))
            mock_tfidf.transform.return_value = mock_sparse_matrix
            
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
    
    def test_api_predict(self):
        """Test the /predict API endpoint."""
        with patch('model_server.predict_sentiment_internal') as mock_predict:
            # Mock the prediction function
            mock_predict.return_value = ('positive', 0.9)
            
            # Create a test client
            client = app.test_client()
            
            # Send a request
            response = client.post('/predict', 
                                  json={'text': 'Test sentence'})
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['sentiment'], 'positive')
            self.assertEqual(data['confidence'], 0.9)
            self.assertIn('latency_ms', data)

    def test_api_predict_batch(self):
        """Test the /predict_batch API endpoint."""
        with patch('model_server.predict_sentiment_internal') as mock_predict:
            # Mock the prediction function
            mock_predict.return_value = ('positive', 0.9)
            
            # Create a test client
            client = app.test_client()
            
            # Send a request
            response = client.post('/predict_batch', 
                                  json={'texts': ['Test 1', 'Test 2']})
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(len(data['results']), 2)
            self.assertEqual(data['count'], 2)
            self.assertIn('latency_ms', data)

    def test_api_metrics(self):
        """Test the /metrics API endpoint."""
        # Create a test client
        client = app.test_client()
        
        # Send a request
        response = client.get('/metrics')
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('total_requests', data)
        self.assertIn('avg_latency_ms', data)

    def test_api_health(self):
        """Test the /health API endpoint."""
        
        client = app.test_client()
        
       
        print("\nAvailable routes:")
        for rule in app.url_map.iter_rules():
            print(f"Route: {rule}, Endpoint: {rule.endpoint}")
        
        
        response = client.get('/health')
        print(f"\nResponse status: {response.status_code}")
        print(f"Response data: {response.data}")
        
        
        self.assertEqual(response.status_code, 200)

    def test_api_routes(self):
        """Debug test to print all available routes."""
        print("\nAvailable routes:")
        for rule in app.url_map.iter_rules():
            print(f"Route: {rule}, Endpoint: {rule.endpoint}")
        
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main() 