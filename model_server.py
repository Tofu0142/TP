import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from collections import deque
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from review_prediction import BertSentimentClassifier, TextBlob
from functools import lru_cache
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
tokenizer = None
device = None
request_times = deque(maxlen=1000)  # Store last 1000 request times
latencies = deque(maxlen=1000)      # Store last 1000 latencies

# Add a global flag to track which model we're using
using_fallback_model = False
fallback_vectorizer = None
fallback_model = None

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

# Model optimization for inference
def optimize_model_for_inference(model):
    model.eval()  # Set to evaluation mode
    
    # Quantize model if possible (reduces precision to speed up inference)
    if hasattr(torch, 'quantization'):
        try:
            # Quantize the model to int8
            model_quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            return model_quantized
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Using original model.")
    
    return model

# Load the model
def load_model(model_path="bert_sentiment_model.pt"):
    global model, tokenizer, device, using_fallback_model, fallback_vectorizer, fallback_model
    
    try:
        # Check if we should use the fallback model
        if os.environ.get('USE_FALLBACK_MODEL') == 'true':
            logger.info("Using fallback model as specified by environment variable")
            # Create a simple fallback model
            texts = [
                "I love this product", "Great experience", "Highly recommended",
                "This is terrible", "Worst purchase ever", "Disappointed",
                "It's okay", "Average performance", "Not bad but not great"
            ]
            labels = [2, 2, 2, 0, 0, 0, 1, 1, 1]  # 0=negative, 1=neutral, 2=positive
            
            # Create a simple model
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.naive_bayes import MultinomialNB
            
            fallback_vectorizer = CountVectorizer()
            X = fallback_vectorizer.fit_transform(texts)
            fallback_model = MultinomialNB()
            fallback_model.fit(X, labels)
            
            using_fallback_model = True
            logger.info("Fallback model created successfully")
            return True
        
        # Regular model loading code
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logger.info("Tokenizer loaded successfully")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            # Try to download if URL is provided
            model_url = os.environ.get('MODEL_URL')
            if model_url:
                logger.info(f"Downloading model from {model_url}")
                try:
                    import requests
                    response = requests.get(model_url)
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Model downloaded to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    raise
            else:
                logger.error(f"Model file not found at {model_path} and no MODEL_URL provided")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize model
        model = BertSentimentClassifier(num_classes=3)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Optimize model for inference
        model = optimize_model_for_inference(model)
        model.to(device)
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
        # Fall back to a simple model
        logger.info("Creating fallback model due to error")
        texts = ["I love this", "I hate this", "It's okay"]
        labels = [2, 0, 1]  # 0=negative, 1=neutral, 2=positive
        
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        
        fallback_vectorizer = CountVectorizer()
        X = fallback_vectorizer.fit_transform(texts)
        fallback_model = MultinomialNB()
        fallback_model.fit(X, labels)
        
        using_fallback_model = True
        logger.info("Fallback model created successfully")
        return True

# Preprocess input for prediction
def preprocess_input(text):
    """
    Preprocess text input for sentiment prediction.
    
    Args:
        text (str): The input text to be analyzed
        
    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Tokenized input IDs
            - attention_mask (torch.Tensor): Attention mask for the input
            - numerical_features (torch.Tensor): Normalized numerical features
            
    This function:
    1. Extracts TextBlob sentiment features
    2. Computes text statistics (word count, character count)
    3. Tokenizes the text using BERT tokenizer
    4. Normalizes numerical features
    """
    # Create DataFrame for processing
    df = pd.DataFrame({'sentence': [text]})
    
    # Extract TextBlob sentiment
    df['body_sentiment'] = df['sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['title_sentiment'] = df['sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Add word count and character count
    df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
    df['char_count'] = df['sentence'].apply(lambda x: len(x))
    
    # Add sentence sentiment
    df['sentiment_score'] = df['body_sentiment']
    
    # Tokenize text
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )
    
    # Extract input IDs and attention masks
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Prepare numerical features
    numerical_features = df[['body_sentiment', 'title_sentiment']].values
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)
    
    # Convert to tensor
    numerical_features = torch.tensor(numerical_features, dtype=torch.float).to(device)
    
    return input_ids, attention_mask, numerical_features

# Make prediction
def predict(input_ids, attention_mask, numerical_features):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, numerical_features)
        _, preds = torch.max(outputs, dim=1)
    
    # Convert numerical predictions to labels
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    prediction = sentiment_labels[preds.item()]
    
    # Get confidence scores
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence = probabilities[0][preds.item()].item()
    
    # Increment the predictions counter
    SENTIMENT_PREDICTIONS.inc()
    
    return prediction, confidence

# Add model caching
@lru_cache(maxsize=1000)
def cached_prediction(text):
    """Cache prediction results for frequently requested texts"""
    input_ids, attention_mask, numerical_features = preprocess_input(text)
    prediction, confidence = predict(input_ids, attention_mask, numerical_features)
    return prediction, confidence

# Define Prometheus metrics
REQUEST_COUNT = Counter('sentiment_requests_total', 'Total number of sentiment analysis requests')
LATENCY = Histogram('sentiment_request_latency_milliseconds', 'Request latency in milliseconds')
ERROR_COUNT = Counter('sentiment_errors_total', 'Total number of errors')
MODEL_CONFIDENCE = Histogram('sentiment_model_confidence', 'Model confidence scores')
SENTIMENT_DISTRIBUTION = Counter('sentiment_results', 'Distribution of sentiment results', ['sentiment'])
SENTIMENT_PREDICTIONS = Counter('sentiment_predictions_total', 'Total number of sentiment predictions')

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Endpoint for sentiment prediction.
    """
    start_time = time.time()
    
    try:
        # Get input data
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Process with appropriate model
        if using_fallback_model:
            # Use the fallback model
            text_vectorized = fallback_vectorizer.transform([text])
            prediction = fallback_model.predict(text_vectorized)[0]
            confidence = float(np.max(fallback_model.predict_proba(text_vectorized)[0]))
            
            # Map prediction to sentiment
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map[prediction]
        else:
            # Use the BERT model
            # Use cached prediction if available
            prediction, confidence = cached_prediction(text)
        
        # Record metrics
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        LATENCY.observe(latency)
        MODEL_CONFIDENCE.observe(confidence)
        SENTIMENT_DISTRIBUTION.labels(sentiment=prediction).inc()
        
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
        logger.error(f"Error in prediction: {str(e)}")
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
        # Process texts in batches of 16
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Preprocess batch
            batch_inputs = [preprocess_input(text) for text in batch_texts]
            
            # Combine into batches
            batch_input_ids = torch.cat([x[0] for x in batch_inputs], dim=0)
            batch_attention_mask = torch.cat([x[1] for x in batch_inputs], dim=0)
            batch_numerical = torch.cat([x[2] for x in batch_inputs], dim=0)
            
            # Batch prediction
            with torch.no_grad():
                outputs = model(batch_input_ids, batch_attention_mask, batch_numerical)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
            
            # Convert to results
            sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
            for j, pred in enumerate(preds):
                idx = i + j
                if idx < len(texts):
                    results.append({
                        "text": texts[idx],
                        "sentiment": sentiment_labels[pred.item()],
                        "confidence": probs[j][pred.item()].item()
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

# Add Prometheus metrics endpoint
@app.route('/prometheus', methods=['GET'])
def prometheus_metrics():
    """
    Endpoint for Prometheus metrics.
    Returns metrics in the Prometheus format.
    """
    try:
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {str(e)}")
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
                <li><code>GET /prometheus</code> - Get Prometheus metrics</li>
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