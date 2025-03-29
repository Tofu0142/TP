FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download BERT model and tokenizer
RUN python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"

# Expose the port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=bert_sentiment_model.pt

# Run the application
CMD ["python", "model_server.py"] 