# Book Review Sentiment Analysis

## Description
This project provides tools for sentiment analysis of text data, including a deployed API for real-time predictions and scripts for analyzing book reviews. The system uses machine learning to determine whether text expresses positive, negative, or neutral sentiment.

## Features
- Sentiment prediction API with health monitoring
- Book review analysis with visualization capabilities
- Combined model using CatBoost, TF-IDF, and Doc2Vec for accurate sentiment classification
- Batch and single text prediction support
- Comprehensive text preprocessing pipeline

## Project Structure
```
sentiment-analysis/
├── model_server.py              # Flask API for sentiment analysis
├── catboost_sentiment_model.cbm # Trained CatBoost model
├── tfidf_vectorizer.pkl         # TF-IDF vectorizer for text features
├── Book_review_analysis.py      # Script for analyzing book reviews
├── review_prediction.py         # Script for training and using the sentiment model
├── test_sentiment_model.py      # Unit tests for the sentiment model
├── integration_test.py          # Integration tests for the API
├── test_model_server.py         # Integration tests for the Flask API
├── load_test.py                 # Performance testing for the API
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore file
└── .github/
    └── workflows/
        └── ci-cd.yml            # CI/CD pipeline configuration
```

## File Descriptions

### `model_server.py`
The main Flask application that serves the sentiment analysis API. It loads the trained CatBoost model and provides endpoints for:
- Health checks (`/health`)
- Single text prediction (`/predict`)
- Batch prediction (`/predict_batch`)
- Metrics monitoring (`/metrics`)

### `review_prediction.py`
A comprehensive script for training and using the sentiment analysis model. This script:
- Loads and processes book review data from JSONL files
- Implements text preprocessing with custom tokenization
- Trains a combined model using CatBoost, TF-IDF, and Doc2Vec
- Performs cross-validation to ensure model quality
- Provides functions for sentiment prediction on new text
- Generates visualizations of model performance

### `catboost_sentiment_model.cbm`
A pre-trained CatBoost classifier model that predicts sentiment based on text features. The model has been trained on a large dataset of labeled text examples.

### `tfidf_vectorizer.pkl`
A serialized TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer that converts raw text into numerical features for the model.

### `Book_review_analysis.py`
A script for analyzing book review data. This script:
- Loads and processes book review data from JSONL files
- Extracts sentiment using TextBlob
- Analyzes common words in negative reviews
- Generates visualizations including:
  - Rating distribution
  - Review length vs. rating
  - Sentiment distribution
  - Word cloud of negative reviews
- Compares verified vs. non-verified purchases

### `test_sentiment_model.py`
Unit tests for the sentiment analysis model to ensure accuracy and reliability.

### `integration_test.py`
Integration tests that verify the API endpoints work correctly and return expected results.

#### `test_model_server.py`
A comprehensive integration test for the Flask API that:
- Tests server startup and shutdown
- Verifies prediction functionality with real requests
- Measures and validates latency (ensuring P99 latency is below 300ms)
- Tests the metrics endpoint
- Tests the health endpoint
- Tests batch prediction functionality

This single test file provides thorough coverage of both functionality and performance aspects of the API.

### `load_test.py`
A performance testing script that evaluates how the API performs under load. This script:
- Simulates multiple concurrent users accessing the API
- Measures response times and success rates
- Calculates key performance metrics (average latency, P95, P99)
- Generates visualizations of latency distribution
- Supports configurable concurrency levels and ramp-up periods

This load testing capability is crucial for ensuring the API can handle production traffic volumes while maintaining acceptable response times.


### `requirements.txt`
Lists all Python dependencies required to run the project, including:
- Flask for the web server
- CatBoost for the ML model
- Pandas and NumPy for data processing
- TextBlob for basic sentiment analysis
- Matplotlib and Seaborn for visualizations
- Gensim for Doc2Vec implementation
- NLTK for text processing

### `Dockerfile`
Configuration for building a Docker container that packages the application and all its dependencies.

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Tofu0142/TP.git
cd TP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources (automatically handled by the scripts, but can be done manually):
```python
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
```

### Usage

#### Training a new model
```bash
python review_prediction.py
```

#### Running the API server
```bash
python model_server.py
```

#### Making predictions
```python
import requests
import json

# Single prediction
response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "This book was absolutely fantastic! I couldn't put it down."}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8080/predict_batch",
    json={"texts": [
        "This book was absolutely fantastic! I couldn't put it down.",
        "The characters were poorly developed and the plot was predictable.",
        "It was an okay read, nothing special but not terrible either."
    ]}
)
print(response.json())
```

### Cloud Deployment

The API is deployed and available on Google Cloud Run at:
[https://sentiment-analysis-438649044905.us-central1.run.app](https://sentiment-analysis-438649044905.us-central1.run.app)

#### Testing the Cloud API

You can test the deployed API using curl:

##### Health Check
```bash
curl https://sentiment-analysis-438649044905.us-central1.run.app/health
```

##### Single Prediction
```bash
curl -X POST \
  https://sentiment-analysis-438649044905.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This book was absolutely fantastic! I could not put it down."}'
```
##### Batch Prediction
```bash
curl -X POST \
  https://sentiment-analysis-438649044905.us-central1.run.app/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This book was absolutely fantastic! I could not put it down.",
      "The characters were poorly developed and the plot was predictable.",
      "It was an okay read, nothing special but not terrible either."
    ]
  }'
```

### Docker Deployment
1. Build the Docker image:
```bash
docker build -t sentiment-analysis .
```

2. Run the container:
```bash
docker run -p 8080:8080 sentiment-analysis
```

## Model Details
The sentiment analysis model combines three types of features:
1. **TF-IDF features**: Captures important words and phrases
2. **Doc2Vec embeddings**: Captures semantic meaning of text
3. **Sentiment scores**: Uses TextBlob to extract basic sentiment polarity

The CatBoost classifier is trained on these combined features to predict sentiment as negative, neutral, or positive.


### Custom Doc2Vec Implementation
While the code references Gensim's Doc2Vec, we've implemented our own custom version for several reasons:
1. **Deployment simplicity**: Avoiding Gensim as a dependency reduces the model's footprint and simplifies deployment
2. **Performance optimization**: Our custom implementation is tailored specifically for sentiment analysis tasks
3. **Integration with preprocessing pipeline**: The custom implementation integrates seamlessly with our text preprocessing steps
4. **Reduced memory usage**: Our implementation uses more efficient data structures for the specific task at hand
5. **Simplified serialization**: Custom implementation allows for easier model serialization and loading

## Acknowledgments
- TextBlob for basic sentiment analysis
- CatBoost for the gradient boosting implementation
