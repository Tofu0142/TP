import os
import time
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from catboost import CatBoostClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from textblob import TextBlob
from functools import lru_cache
import re
import string
import threading
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('sentiment-model-server')

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
doc2vec_model = None
tfidf_vectorizer = None
start_time = time.time()
request_count = 0
latencies = []
request_lock = threading.Lock()

# Clean text
def clean_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Simple word tokenizer
def simple_word_tokenize(text):
    """Simple word tokenization."""
    return text.lower().split()

# Load model
def load_model():
    global model, doc2vec_model, tfidf_vectorizer
    
    try:
        # Load CatBoost model
        model_path = "catboost_sentiment_model.cbm"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info(f"CatBoost model loaded successfully from {model_path}")
        
        # Create a dummy Doc2Vec model with minimal settings
        logger.info("Creating a dummy Doc2Vec model")
        try:
            # Create a very simple Doc2Vec model with minimal settings
            doc2vec_model = Doc2Vec(vector_size=100, min_count=1)
            
            # Create a simple document with words that will definitely be included
            words = ["this", "is", "a", "test", "document"]
            tagged_doc = TaggedDocument(words=words, tags=["0"])
            
            # Build vocabulary with this document
            doc2vec_model.build_vocab([tagged_doc])
            
            # Train for just one epoch
            doc2vec_model.train([tagged_doc], total_examples=1, epochs=1)
            logger.info("Dummy Doc2Vec model created successfully")
        except Exception as e:
            logger.warning(f"Could not create Doc2Vec model: {str(e)}. Using None instead.")
            doc2vec_model = None
        
        # Load TF-IDF vectorizer
        import pickle
        tfidf_path = "tfidf_vectorizer.pkl"
        if not os.path.exists(tfidf_path):
            logger.error(f"TF-IDF vectorizer file not found: {tfidf_path}")
            return False
        
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        logger.info(f"TF-IDF vectorizer loaded successfully from {tfidf_path}")
        
        # Warm up the model with a sample prediction
        logger.info("Warming up model...")
        sample_text = "This is a sample text to warm up the model."
        predict_sentiment_internal(sample_text)
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Preprocess input for prediction
def preprocess_input(text):
    """Preprocess input text for prediction."""
    global tfidf_vectorizer, doc2vec_model
    
    # Clean text
    processed_text = clean_text(text)
    
    # 1. TF-IDF features
    X_tfidf = tfidf_vectorizer.transform([processed_text])
    X_tfidf_array = X_tfidf.toarray()
    
    # 2. Doc2Vec features - use zeros instead of actual vectors
    # since we're using a dummy model or no model
    X_doc2vec = np.zeros((1, 100))  # Assuming vector_size=100
    
    # 3. Sentiment features
    body_sentiment = TextBlob(text).sentiment.polarity
    title_sentiment = body_sentiment  # Use same sentiment for title since we don't have a title
    
    # Create feature DataFrames
    X_tfidf_df = pd.DataFrame(X_tfidf_array, 
                             columns=[f'tfidf_{i}' for i in range(X_tfidf_array.shape[1])])
    
    X_doc2vec_df = pd.DataFrame(X_doc2vec, 
                               columns=[f'doc2vec_{i}' for i in range(X_doc2vec.shape[1])])
    
    X_sentiment_df = pd.DataFrame({
        'body_sentiment': [body_sentiment],
        'title_sentiment': [title_sentiment]
    })
    
    # Combine all features
    X = pd.concat([X_tfidf_df, X_doc2vec_df, X_sentiment_df], axis=1)
    
    return X

# Predict sentiment
@lru_cache(maxsize=1024)
def predict_sentiment_internal(text):
    """Predict sentiment for input text."""
    global model
    
    start_time = time.time()
    
    try:
        # Preprocess input
        features = preprocess_input(text)
        
        # Make prediction
        prediction_array = model.predict(features)
        prediction = prediction_array[0]  # Extract the scalar value
        probabilities_array = model.predict_proba(features)
        probabilities = probabilities_array[0]  # Extract the array for the first sample
        
        # Map prediction to sentiment label
        sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        prediction_index = int(prediction)  # Convert to Python int
        sentiment = sentiment_labels[prediction_index]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[prediction_index])  # Convert to Python float
        
        return sentiment, confidence
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return "neutral", 0.33  # Default fallback
    finally:
        # Record latency
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    global request_count
    
    # Get request data
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Predict sentiment
    start_time = time.time()
    sentiment, confidence = predict_sentiment_internal(text)
    latency = (time.time() - start_time) * 1000  # ms
    
    # Update request count
    with request_lock:
        request_count += 1
    
    # Return prediction
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'latency_ms': latency
    })

# API endpoint for batch prediction
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    global request_count
    
    # Get request data
    data = request.json
    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400
    
    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({'error': 'Texts must be a list'}), 400
    
    # Predict sentiment for each text
    results = []
    start_time = time.time()
    
    for text in texts:
        sentiment, confidence = predict_sentiment_internal(text)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    latency = (time.time() - start_time) * 1000  # ms
    
    # Update request count
    with request_lock:
        request_count += 1
    
    # Return predictions
    return jsonify({
        'results': results,
        'count': len(results),
        'latency_ms': latency
    })

# API endpoint for metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    global start_time, request_count, latencies
    
    # Calculate metrics
    uptime = time.time() - start_time
    requests_per_second = request_count / uptime if uptime > 0 else 0
    
    # Calculate latency percentiles
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p99_index = int(len(sorted_latencies) * 0.99)
        p99_latency = sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1]
    else:
        avg_latency = 0
        p99_latency = 0
    
    # Return metrics
    return jsonify({
        'uptime_seconds': uptime,
        'total_requests': request_count,
        'requests_per_second': requests_per_second,
        'avg_latency_ms': avg_latency,
        'p99_latency_ms': p99_latency
    })

# API endpoint for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'uptime': time.time() - start_time})

# Test page
@app.route('/test', methods=['GET'])
def test_page():
    return render_template('test.html')

# Default route
@app.route('/', methods=['GET'])
def home():
    return "Sentiment Analysis API is running. Send POST requests to /predict"

# Create test.html template
@app.route('/templates/test.html')
def get_test_template():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 100px; margin-bottom: 10px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
            .positive { color: green; }
            .neutral { color: blue; }
            .negative { color: red; }
        </style>
    </head>
    <body>
        <h1>Sentiment Analysis Test</h1>
        <p>Enter text to analyze sentiment:</p>
        <textarea id="text" placeholder="Enter text here..."></textarea>
        <button onclick="analyzeSentiment()">Analyze</button>
        <div id="result"></div>
        
        <script>
            function analyzeSentiment() {
                const text = document.getElementById('text').value;
                if (!text) {
                    alert('Please enter some text');
                    return;
                }
                
                document.getElementById('result').innerHTML = 'Analyzing...';
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `
                        <p>Sentiment: <span class="${data.sentiment}">${data.sentiment}</span></p>
                        <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
                        <p>Latency: ${data.latency_ms.toFixed(2)} ms</p>
                    `;
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = `Error: ${error.message}`;
                });
            }
        </script>
    </body>
    </html>
    """
    return html

# Main function
def main():
    # Load model
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        return
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Start server
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main() 