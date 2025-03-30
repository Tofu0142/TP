FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY model_server.py .
COPY catboost_sentiment_model.cbm .
COPY tfidf_vectorizer.pkl .

# Expose the port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "model_server.py"]