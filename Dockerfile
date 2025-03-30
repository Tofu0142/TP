# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch first (much smaller download)
RUN pip install --no-cache-dir --timeout=100 --retries=10 \
    torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir --timeout=100 --retries=10 -r requirements.txt

# Download BERT tokenizer and cache it in the virtual environment
RUN python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=bert_sentiment_model.pt
ARG USE_FALLBACK_MODEL=false
ENV USE_FALLBACK_MODEL=${USE_FALLBACK_MODEL}
ARG MODEL_URL=""
ENV MODEL_URL=${MODEL_URL}

# Copy application code
COPY . .

# Download model if URL is provided and not using fallback
RUN if [ "$USE_FALLBACK_MODEL" = "false" ] && [ -n "$MODEL_URL" ]; then \
        echo "Downloading model from ${MODEL_URL}"; \
        python -c "import requests; open('${MODEL_PATH}', 'wb').write(requests.get('${MODEL_URL}', timeout=60).content)"; \
    else \
        echo "Using fallback model or no model URL provided"; \
    fi

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "model_server.py"] 