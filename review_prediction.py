import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import re
from collections import Counter
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set up NLTK data path and download resources
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download {resource}: {e}")

# Call this at the beginning to ensure resources are available
download_nltk_resources()

# Load the JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return data

# Enhanced text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Normalization: Convert to lowercase
    text = text.lower()
    
    # Noise Removal: Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)  # Simple HTML tag removal
    
    # Remove HTML entities like &nbsp;
    text = re.sub(r'&\w+;', ' ', text)
    
    # Remove ampersand character
    text = text.replace('&', ' ')
    
    # Remove special characters but keep alphanumeric words
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback if word_tokenize fails
        tokens = text.split()
    
    # Noise Removal: Remove stop words
    try:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words]
    except LookupError:
        # Fallback if stopwords fails
        common_stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                            'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                            'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                            'don', 'should', 'now'}
        filtered_tokens = [w for w in tokens if w not in common_stop_words]
    
    # Rejoin tokens
    processed_text = " ".join(filtered_tokens)
    
    # Remove extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

# Process the data and split reviews into sentences
def process_data_sentences(data):
    sentences = []
    titles = []
    ratings = []
    body_sentiments = []  # List to store body sentiment for each sentence
    title_sentiments = []  # List to store title sentiment for each sentence
    
    for review in data:
        try:
            # Extract review text, title, and rating
            text = review['text']
            title = review['title']
            rating = review['rating']
            
            # Calculate full review sentiment (body sentiment)
            body_sentiment = TextBlob(text).sentiment.polarity
            
            # Calculate title sentiment
            title_sentiment = TextBlob(title).sentiment.polarity
            
            # Clean the text first
            cleaned_text = preprocess_text(text)
            
            # Split review into sentences - with fallback method
            try:
                review_sentences = sent_tokenize(cleaned_text)
            except LookupError:
                # Simple fallback sentence tokenization
                review_sentences = re.split(r'[.!?]+', cleaned_text)
            
            # Add each sentence with its title and rating
            for sentence in review_sentences:
                # Skip very short sentences or sentences with just punctuation
                if len(sentence.strip()) > 5 and any(c.isalpha() for c in sentence):
                    sentences.append(sentence)
                    titles.append(title)
                    ratings.append(rating)
                    body_sentiments.append(body_sentiment)  # Add body sentiment for this sentence
                    title_sentiments.append(title_sentiment)  # Add title sentiment for this sentence
        except KeyError:
            continue  # Skip reviews with missing fields
    
    # Create DataFrame
    df = pd.DataFrame({
        'sentence': sentences,
        'title': titles,
        'rating': ratings,
        'body_sentiment': body_sentiments,    # Now correctly using the lists
        'title_sentiment': title_sentiments
    })
    
    # Create sentiment labels based on rating
    df['sentiment'] = pd.cut(
        df['rating'],
        bins=[0, 2.5, 3.5, 5.1],
        labels=['negative', 'neutral', 'positive']
    )
    
    return df

# Clean text function (updated to use the enhanced preprocessing)
def clean_text(text):
    return preprocess_text(text)

# Custom transformer to extract features from text columns
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.field].apply(clean_text).values

# Train sentiment classifier
def train_sentiment_classifier(df):
    # Prepare data
    X = df[['sentence', 'title', 'body_sentiment', 'title_sentiment']]
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create feature extraction pipeline
    sentence_pipeline = Pipeline([
        ('selector', TextSelector('sentence')),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8))
    ])
    
    title_pipeline = Pipeline([
        ('selector', TextSelector('title')),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8))
    ])

    # Combine features
    
    features = FeatureUnion([
        ('sentence_features', sentence_pipeline),
        ('title_features', title_pipeline),
        ('body_sentiment_features', ),
    ])
    
    # Create full pipeline
    pipeline = Pipeline([
        ('features', features),
        ('classifier', LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42,
            C=1.0
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    
    return pipeline, X_test, y_test, y_pred, accuracy, report, conf_matrix

# Analyze most informative features
def analyze_features(pipeline, classes):
    # Get feature names from both pipelines
    sentence_features = pipeline.named_steps['features'].transformer_list[0][1].named_steps['tfidf'].get_feature_names_out()
    title_features = pipeline.named_steps['features'].transformer_list[1][1].named_steps['tfidf'].get_feature_names_out()
    
    # Add prefixes to distinguish features
    sentence_features = ['sentence_' + f for f in sentence_features]
    title_features = ['title_' + f for f in title_features]
    
    # Combine feature names
    feature_names = np.concatenate([sentence_features, title_features])
    
    # Get coefficients
    coefficients = pipeline.named_steps['classifier'].coef_
    
    # Create dictionary to store top features for each class
    feature_importance = {}
    
    for i, sentiment in enumerate(classes):
        # Get coefficients for this class
        class_coef = coefficients[i]
        
        # Sort features by coefficient
        sorted_features = sorted(zip(feature_names, class_coef), key=lambda x: -abs(x[1]))
        
        # Store in dictionary
        feature_importance[sentiment] = sorted_features
    
    return feature_importance

# Visualize results
def visualize_results(df, y_test, y_pred, conf_matrix):
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Sentiment distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Plot 2: Rating distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['rating'], bins=5, kde=True)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Plot 3: Confusion matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 4: Accuracy by sentiment
    plt.subplot(2, 2, 4)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracies = {k: v['f1-score'] for k, v in report_dict.items() 
                 if k in ['negative', 'neutral', 'positive']}
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('F1-Score by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('F1-Score')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png')
    plt.close()

# Predict sentiment for new reviews
def predict_sentiment(pipeline, reviews):
    all_sentences = []
    clean_sentences = []
    
    for review in reviews:
        # Split review into sentences
        sentences = sent_tokenize(review)
        
        # Process each sentence
        for sentence in sentences:
            if len(sentence.strip()) > 5 and any(c.isalpha() for c in sentence):
                all_sentences.append((review, sentence))
                # Create a DataFrame with sentence and title (using sentence as title for demonstration)
                clean_sentences.append(pd.DataFrame({
                    'sentence': [sentence],
                    'title': [sentence]  # Using sentence as title for simplicity
                }))
    
    # Make predictions
    predictions = []
    for df in clean_sentences:
        predictions.append(pipeline.predict(df)[0])
    
    # Organize results
    results = []
    for i, (review, sentence) in enumerate(all_sentences):
        results.append({
            'review': review,
            'sentence': sentence,
            'sentiment': predictions[i]
        })
    
    return results

# BERT Sentiment Classifier model - simplified version without numerical features
class BertSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        # BERT output size (768)
        bert_output_size = 768
        
        # Fully connected layers
        self.fc1 = nn.Linear(bert_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, numerical_features=None):
        # Process text through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        
        # Process BERT output
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Prepare data for BERT - simplified version without numerical features
def prepare_bert_data(df, max_length=128):
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare text inputs
    sentences = df['sentence'].values
    
    # Tokenize all sentences
    encoded_data = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    # Extract input IDs and attention masks
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    
    # Create a dummy tensor for numerical features (for compatibility)
    numerical_features = torch.zeros((len(df), 1), dtype=torch.float)
    
    # Prepare labels
    labels = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, numerical_features, labels

# Train BERT model
def train_bert_model(df, batch_size=16, epochs=3):
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    
    # Prepare data
    train_inputs, train_masks, train_numerical, train_labels = prepare_bert_data(train_df)
    val_inputs, val_masks, val_numerical, val_labels = prepare_bert_data(val_df)
    
    # Create DataLoader for training
    train_data = TensorDataset(train_inputs, train_masks, train_numerical, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create DataLoader for validation
    val_data = TensorDataset(val_inputs, val_masks, val_numerical, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = BertSentimentClassifier(num_classes=3)
    model.to(device)
    
    # Set optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Clear gradients
            optimizer.zero_grad()
            
            # Unpack batch
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, numerical_features, labels = batch
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_accuracy = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, numerical_features, labels = batch
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                
                # Calculate accuracy
                val_accuracy += (preds == labels).sum().item()
                
                # Store predictions and true labels
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())
        
        val_accuracy /= len(val_df)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'bert_sentiment_model.pt')
            print(f'Model saved with accuracy: {val_accuracy:.4f}')
        
        # Print classification report
        print('\nClassification Report:')
        target_names = ['negative', 'neutral', 'positive']
        print(classification_report(val_true, val_preds, target_names=target_names))
    
    # Load best model
    model.load_state_dict(torch.load('bert_sentiment_model.pt'))
    
    # Convert numerical predictions back to labels
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    val_preds_labels = [sentiment_labels[pred] for pred in val_preds]
    val_true_labels = [sentiment_labels[label] for label in val_true]
    
    return model, val_true_labels, val_preds_labels

# Predict sentiment using BERT
def predict_with_bert(model, sentences, batch_size=16):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize sentences
    encoded_data = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )
    
    # Extract input IDs and attention masks
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    
    # Create a dummy tensor for numerical features (for compatibility)
    numerical_features = torch.zeros((len(sentences), 1), dtype=torch.float)
    
    # Create DataLoader
    prediction_data = TensorDataset(input_ids, attention_masks, numerical_features)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prediction
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, numerical_features = batch
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
    
    # Convert numerical predictions to labels
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = [sentiment_labels[pred] for pred in predictions]
    
    return predictions

# Main function
def main():
    # Download necessary NLTK data
    #download_nltk_resources()
    
    # Load the data
    file_path = 'Books_10k.jsonl'
    print("Loading data...")
    data = load_jsonl(file_path)
    
    print("Processing data and splitting reviews into sentences...")
    df = process_data_sentences(data)
    
    print(f"Total sentences extracted: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train BERT model
    print("Training BERT model...")
    model, y_true, y_pred = train_bert_model(df, batch_size=16, epochs=3)
    
    print(f"Model accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    print(conf_matrix)
    
    print("\nVisualizing results...")
    visualize_results(df, y_true, y_pred)
    
    # Example of predicting sentiment for new reviews
    sample_reviews = [
        "This book was absolutely fantastic! I couldn't put it down.",
        "The characters were poorly developed and the plot was predictable.",
        "It was an okay read, nothing special but not terrible either."
    ]
    
    print("\nPredicting sentiment for sample reviews:")
    predictions = predict_with_bert(model, sample_reviews)
    
    for i, review in enumerate(sample_reviews):
        print(f"Review: {review}")
        print(f"Predicted sentiment: {predictions[i]}")
        print("---")

# if __name__ == "__main__":
#     main()