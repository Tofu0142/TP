import os
import time
import json
import logging
import numpy as np
import pandas as pd
import re
from flask import Flask, request, jsonify
from collections import deque
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from textblob import TextBlob
from functools import lru_cache
from catboost import CatBoostClassifier
from gensim.models.doc2vec import Doc2Vec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment-model-server")

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
doc2vec_model = None
tfidf_vectorizer = None
request_times = deque(maxlen=1000)  # Store last 1000 request times
latencies = deque(maxlen=1000)      # Store last 1000 latencies

# Simple word tokenization function
def simple_word_tokenize(text):
    # First replace punctuation with spaces
    for punct in '.,!?;:()[]{}"\'-':
        text = text.replace(punct, ' ')
    # Split by spaces
    return [word for word in text.split() if word]

# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Normalization: convert to lowercase
    text = text.lower()
    
    # Noise removal: delete HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove HTML entities like &nbsp;
    text = re.sub(r'&\w+;', ' ', text)
    
    # Remove & character
    text = text.replace('&', ' ')
    
    # Remove special characters but keep alphanumeric words
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Performance monitoring
class PerformanceMonitor:
    def __init__(self, window_size=1000):
        self.latencies = deque(maxlen=window_size)
        self.request_times = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
    def add_latency(self, latency_ms):
        with self.lock:
            self.latencies.append(latency_ms)
            self.request_times.append(datetime.now())
    
    def get_p99_latency(self):
        with self.lock:
            if not self.latencies:
                return 0
            return np.percentile(self.latencies, 99)
    
    def get_avg_latency(self):
        with self.lock:
            if not self.latencies:
                return 0
            return np.mean(self.latencies)
    
    def get_request_rate(self, seconds=60):
        with self.lock:
            if not self.request_times:
                return 0
            now = datetime.now()
            count = sum(1 for t in self.request_times if (now - t).total_seconds() <= seconds)
            return count / seconds  # requests per second
    
    def generate_report(self):
        return {
            "p99_latency_ms": self.get_p99_latency(),
            "avg_latency_ms": self.get_avg_latency(),
            "requests_per_second": self.get_request_rate(),
            "total_requests": len(self.latencies)
        }
    
    def get_latency_data(self):
        """Return the latency data for plotting instead of creating plots directly"""
        with self.lock:
            if len(self.latencies) < 10:
                return None
            
            data = {
                "latencies": list(self.latencies),
                "p99": self.get_p99_latency(),
                "avg": self.get_avg_latency()
            }
            return data

# Initialize performance monitor
monitor = PerformanceMonitor()

# Load the model
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
        
        # Load Doc2Vec model
        doc2vec_path = "doc2vec_model"
        if not os.path.exists(doc2vec_path):
            logger.error(f"Doc2Vec model file not found: {doc2vec_path}")
            return False
        
        doc2vec_model = Doc2Vec.load(doc2vec_path)
        logger.info(f"Doc2Vec model loaded successfully from {doc2vec_path}")
        
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
    
    # 2. Doc2Vec features
    X_doc2vec = np.array([doc2vec_model.infer_vector(simple_word_tokenize(processed_text))])
    
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
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map prediction to sentiment label
        sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_labels[int(prediction)]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[int(prediction)])
        
        return sentiment, confidence
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return "neutral", 0.33  # Default fallback
    finally:
        # Record latency
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)

# Add model caching
@lru_cache(maxsize=1000)
def cached_prediction(text):
    """Cache prediction results for frequently requested texts"""
    prediction, confidence = predict_sentiment_internal(text)
    return prediction, confidence

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    start_time = time.time()
    
    # Debug info
    logger.info("Received prediction request")
    
    # Get request data
    if not request.is_json:
        logger.error("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    logger.info(f"Request data: {data}")
    
    if 'text' not in data:
        logger.error("Missing 'text' field in request")
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data['text']
    logger.info(f"Processing text: {text[:50]}...")
    
    try:
        # Use cached prediction if available
        prediction, confidence = cached_prediction(text)
        
        # Record metrics
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Add to performance monitor
        monitor.add_latency(latency)
        
        # Log request
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Latency: {latency:.2f}ms")
        
        # Return result
        return jsonify({
            "text": text,
            "sentiment": prediction,
            "confidence": confidence,
            "latency_ms": latency
        })
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

# API endpoint for monitoring
@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Generate performance report
    report = monitor.generate_report()
    
    # Add histogram data as text representation
    latency_data = monitor.get_latency_data()
    if latency_data and len(latency_data["latencies"]) > 0:
        # Add percentile information
        percentiles = {
            "p50": np.percentile(latency_data["latencies"], 50),
            "p90": np.percentile(latency_data["latencies"], 90),
            "p95": np.percentile(latency_data["latencies"], 95),
            "p99": latency_data["p99"],
            "min": min(latency_data["latencies"]),
            "max": max(latency_data["latencies"]),
            "avg": latency_data["avg"]
        }
        report["latency_percentiles"] = percentiles
        
        # Create a simple text-based histogram representation
        hist, bin_edges = np.histogram(latency_data["latencies"], bins=10)
        histogram_data = []
        for i in range(len(hist)):
            histogram_data.append({
                "bin_start": bin_edges[i],
                "bin_end": bin_edges[i+1],
                "count": int(hist[i])
            })
        report["latency_histogram"] = histogram_data
    
    return jsonify(report)

# Background thread for periodic monitoring
def monitoring_thread():
    while True:
        try:
            # Log performance metrics every minute
            report = monitor.generate_report()
            logger.info(f"Performance metrics: {json.dumps(report)}")
            
            # Check if p99 latency exceeds target
            if report["p99_latency_ms"] > 300:
                logger.warning(f"P99 latency ({report['p99_latency_ms']:.2f}ms) exceeds target (300ms)")
            
            # Sleep for 60 seconds
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in monitoring thread: {e}")
            time.sleep(60)

# Add batch prediction endpoint
@app.route('/batch_predict', methods=['POST'])
def batch_predict_sentiment():
    start_time = time.time()
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({"error": "Request must contain a 'texts' array"}), 400
    
    texts = data['texts']
    results = []
    
    try:
        # Process each text
        for text in texts:
            prediction, confidence = predict_sentiment_internal(text)
            results.append({
                "text": text,
                "sentiment": prediction,
                "confidence": confidence
            })
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000
        
        return jsonify({
            "results": results,
            "batch_size": len(texts),
            "latency_ms": latency
        })
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Add a route for the root URL
@app.route('/', methods=['GET'])
def index():
    """Serve a simple welcome page or API documentation."""
    return """
    <html>
        <head>
            <title>Model Server API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                ul { margin-bottom: 20px; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Model Server API</h1>
            <p>Welcome to the Model Server API. Available endpoints:</p>
            <ul>
                <li><code>POST /predict</code> - Make a single prediction</li>
                <li><code>POST /batch_predict</code> - Make batch predictions</li>
                <li><code>GET /metrics</code> - Get model metrics</li>
                <li><code>GET /metrics_view</code> - Get metrics in a readable format</li>
                <li><code>GET /test</code> - Test the sentiment analysis endpoint</li>
            </ul>
            <p>For API documentation, please refer to the project documentation.</p>
        </body>
    </html>
    """

# Add a metrics visualization page
@app.route('/metrics_view', methods=['GET'])
def metrics_view():
    """Serve a page that displays metrics in a readable format."""
    return """
    <html>
        <head>
            <title>Model Metrics</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1, h2 { color: #333; }
                .metric { margin-bottom: 10px; }
                .metric-name { font-weight: bold; }
                .metric-value { margin-left: 10px; }
                #metrics { margin-top: 20px; }
                button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <h1>Model Metrics</h1>
            <button onclick="refreshMetrics()">Refresh Metrics</button>
            <div id="metrics">Loading metrics...</div>
            
            <script>
                function refreshMetrics() {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            let html = '<h2>Performance Metrics</h2>';
                            
                            // Basic metrics
                            html += `<div class="metric"><span class="metric-name">Total Requests:</span><span class="metric-value">${data.total_requests}</span></div>`;
                            html += `<div class="metric"><span class="metric-name">Average Latency:</span><span class="metric-value">${data.avg_latency_ms.toFixed(2)}ms</span></div>`;
                            html += `<div class="metric"><span class="metric-name">P99 Latency:</span><span class="metric-value">${data.p99_latency_ms.toFixed(2)}ms</span></div>`;
                            html += `<div class="metric"><span class="metric-name">Requests Per Second:</span><span class="metric-value">${data.requests_per_second.toFixed(2)}</span></div>`;
                            
                            // Percentiles if available
                            if (data.latency_percentiles) {
                                html += '<h2>Latency Percentiles</h2>';
                                for (const [key, value] of Object.entries(data.latency_percentiles)) {
                                    html += `<div class="metric"><span class="metric-name">${key}:</span><span class="metric-value">${value.toFixed(2)}ms</span></div>`;
                                }
                            }
                            
                            // Histogram if available
                            if (data.latency_histogram) {
                                html += '<h2>Latency Histogram</h2>';
                                html += '<table border="1" cellpadding="5" style="border-collapse: collapse;">';
                                html += '<tr><th>Range (ms)</th><th>Count</th><th>Visualization</th></tr>';
                                
                                const maxCount = Math.max(...data.latency_histogram.map(bin => bin.count));
                                
                                for (const bin of data.latency_histogram) {
                                    const barWidth = bin.count > 0 ? (bin.count / maxCount * 200) : 0;
                                    html += `<tr>
                                        <td>${bin.bin_start.toFixed(1)} - ${bin.bin_end.toFixed(1)}</td>
                                        <td>${bin.count}</td>
                                        <td><div style="background-color: #4CAF50; width: ${barWidth}px; height: 20px;"></div></td>
                                    </tr>`;
                                }
                                
                                html += '</table>';
                            }
                            
                            document.getElementById('metrics').innerHTML = html;
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            document.getElementById('metrics').innerHTML = 'Error loading metrics. See console for details.';
                        });
                }
                
                // Load metrics on page load
                refreshMetrics();
                
                // Refresh metrics every 10 seconds
                setInterval(refreshMetrics, 10000);
            </script>
        </body>
    </html>
    """

# Add a test page for the predict endpoint
@app.route('/test', methods=['GET'])
def test_page():
    """Serve a simple test page for the predict endpoint."""
    return """
    <html>
        <head>
            <title>Test Sentiment Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                textarea { width: 100%; height: 100px; margin-bottom: 10px; }
                button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; display: none; }
            </style>
        </head>
        <body>
            <h1>Test Sentiment Analysis</h1>
            <textarea id="text" placeholder="Enter text to analyze..."></textarea>
            <button onclick="analyzeSentiment()">Analyze</button>
            <div id="result"></div>
            
            <script>
                function analyzeSentiment() {
                    const text = document.getElementById('text').value;
                    if (!text) {
                        alert('Please enter some text');
                        return;
                    }
                    
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({text: text}),
                    })
                    .then(response => response.json())
                    .then(data => {
                        const resultDiv = document.getElementById('result');
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = `
                            <h3>Result:</h3>
                            <p>Sentiment: <strong>${data.sentiment}</strong></p>
                            <p>Confidence: <strong>${(data.confidence * 100).toFixed(2)}%</strong></p>
                            <p>Latency: <strong>${data.latency_ms.toFixed(2)}ms</strong></p>
                        `;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error analyzing sentiment. See console for details.');
                    });
                }
            </script>
        </body>
    </html>
    """

# Start the server
if __name__ == '__main__':
    # Load the model
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitoring_thread, daemon=True)
    monitor_thread.start()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Start Flask server
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True) 