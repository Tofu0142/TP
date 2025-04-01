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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from catboost import CatBoostClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set NLTK data path and download resources
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

# Call this function at the beginning to ensure resources are available
download_nltk_resources()

# Load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return data

# Simple word tokenization function to replace word_tokenize
def simple_word_tokenize(text):
    # First replace punctuation with spaces
    for punct in '.,!?;:()[]{}"\'-':
        text = text.replace(punct, ' ')
    # Split by spaces
    return [word for word in text.split() if word]

# Simple sentence tokenization function to replace sent_tokenize
def simple_sent_tokenize(text):
    # Split by sentence ending punctuation
    sentences = []
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if sent.strip():
            sentences.append(sent.strip())
    return sentences if sentences else [text]

# Modified preprocess_text function using simple tokenization
def preprocess_text(text):
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
    
    # Tokenization - use simple tokenization instead of NLTK
    tokens = simple_word_tokenize(text)
    
    # Noise removal: remove stopwords
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
    
    # Remove extra spaces
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

# Modified process_data_sentences function using simple sentence tokenization
def process_data_sentences(data):
    sentences = []
    titles = []
    ratings = []
    body_sentiments = []
    title_sentiments = []
    
    for review in data:
        try:
            # Extract review text, title and rating
            text = review['text']
            title = review['title']
            rating = review['rating']
            
            # Calculate full review sentiment (body sentiment)
            body_sentiment = TextBlob(text).sentiment.polarity
            
            # Calculate title sentiment
            title_sentiment = TextBlob(title).sentiment.polarity
            
            # First clean the text
            cleaned_text = preprocess_text(text)
            
            # Split review into sentences - use simple sentence tokenization
            review_sentences = simple_sent_tokenize(cleaned_text)
            
            # Add each sentence with its title and rating
            for sentence in review_sentences:
                # Skip very short sentences or sentences with only punctuation
                if len(sentence.strip()) > 5 and any(c.isalpha() for c in sentence):
                    sentences.append(sentence)
                    titles.append(title)
                    ratings.append(rating)
                    body_sentiments.append(body_sentiment)
                    title_sentiments.append(title_sentiment)
        except KeyError:
            continue
    
    # Create DataFrame
    df = pd.DataFrame({
        'sentence': sentences,
        'title': titles,
        'rating': ratings,
        'body_sentiment': body_sentiments,
        'title_sentiment': title_sentiments
    })
    
    # Create sentiment labels based on ratings
    df['sentiment'] = pd.cut(
        df['rating'],
        bins=[0, 2.5, 3.5, 5.1],
        labels=['negative', 'neutral', 'positive']
    )
    
    return df

# Text cleaning function (updated to use enhanced preprocessing)
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

# Modified train_doc2vec_model function using simple tokenization
def train_doc2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=20):
    """Train a Doc2Vec model on the given sentences."""
    # Tokenize sentences
    tokenized_sentences = [simple_word_tokenize(s) for s in sentences]
    
    # Create tagged documents
    tagged_data = [TaggedDocument(words=tokens, tags=[str(i)]) for i, tokens in enumerate(tokenized_sentences)]
    
    # Initialize and train Doc2Vec model
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Create directory if it doesn't exist
    os.makedirs('doc2vec_model', exist_ok=True)
    
    # Save model using pickle
    with open('doc2vec_model/doc2vec.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Modified get_doc2vec_vectors function using simple tokenization
def get_doc2vec_vectors(model, sentences):
    vectors = []
    for sentence in sentences:
        # Preprocess text
        processed = clean_text(sentence)
        # Get vector
        vector = model.infer_vector(simple_word_tokenize(processed))
        vectors.append(vector)
    
    return np.array(vectors)

# Visualize results
def visualize_results(df, y_test, y_pred, conf_matrix):
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Sentiment distribution
    plt.subplot(1, 3, 1)
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    
    # Plot 2: Confusion matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 3: F1 score by sentiment
    plt.subplot(1, 3, 3)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracies = {k: v['f1-score'] for k, v in report_dict.items() 
                 if k in ['negative', 'neutral', 'positive']}
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('F1 Score by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png')
    plt.close()

# Train model using CatBoost, TF-IDF and Doc2Vec with cross-validation
def train_combined_model(df, iterations=100, learning_rate=0.1, cv_folds=5):
    print("Preparing features...")
    
    # Prepare text data
    sentences = df['sentence'].values
    processed_sentences = [clean_text(s) for s in sentences]
    
    # 1. TF-IDF features
    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(processed_sentences)
    
    # 2. Doc2Vec features
    print("Training Doc2Vec model...")
    doc2vec_model = train_doc2vec_model(sentences)
    
    print("Extracting Doc2Vec features...")
    X_doc2vec = get_doc2vec_vectors(doc2vec_model, sentences)
    
    # 3. Sentiment features
    X_sentiment = df[['body_sentiment', 'title_sentiment']].values
    
    # Convert TF-IDF features to DataFrame
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), 
                             columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])
    
    # Convert Doc2Vec features to DataFrame
    X_doc2vec_df = pd.DataFrame(X_doc2vec, 
                               columns=[f'doc2vec_{i}' for i in range(X_doc2vec.shape[1])])
    
    # Convert sentiment features to DataFrame
    X_sentiment_df = pd.DataFrame(X_sentiment, 
                                 columns=['body_sentiment', 'title_sentiment'])
    
    # Combine all features
    print("Combining features...")
    X = pd.concat([X_tfidf_df, X_doc2vec_df, X_sentiment_df], axis=1)
    
    # Prepare labels
    y = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    
    # Perform cross-validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    print(f"Performing {cv_folds}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    fold = 1
    
    # For final model and evaluation
    final_y_test = []
    final_y_pred = []
    
    for train_index, test_index in skf.split(X, y):
        print(f"\nTraining fold {fold}/{cv_folds}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize and train CatBoost model
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            loss_function='MultiClass',
            random_seed=42,
            verbose=100
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores.append(accuracy)
        
        print(f"Fold {fold} accuracy: {accuracy:.4f}")
        
        # Save results for final evaluation
        final_y_test.extend(y_test)
        final_y_pred.extend(y_pred)
        
        fold += 1
    
    # Print cross-validation results
    print(f"\nCross-validation complete!")
    print(f"Mean accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard deviation: {np.std(cv_scores):.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        loss_function='MultiClass',
        random_seed=42,
        verbose=100
    )
    
    final_model.fit(X, y, verbose=False)
    
    # Save model and vectorizers
    final_model.save_model('catboost_sentiment_model.cbm')
    
    # Save TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Convert numeric predictions to labels for final evaluation
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    y_pred_labels = [sentiment_labels[int(pred)] for pred in final_y_pred]
    y_test_labels = [sentiment_labels[int(label)] for label in final_y_test]
    
    return final_model, doc2vec_model, tfidf_vectorizer, y_test_labels, y_pred_labels

# Predict sentiment using combined model
def predict_with_combined_model(model, doc2vec_model, tfidf_vectorizer, sentences):
    # Preprocess text
    processed_sentences = [clean_text(sentence) for sentence in sentences]
    
    # 1. Extract TF-IDF features
    X_tfidf = tfidf_vectorizer.transform(processed_sentences)
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), 
                             columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])
    
    # 2. Extract Doc2Vec features
    X_doc2vec = get_doc2vec_vectors(doc2vec_model, sentences)
    X_doc2vec_df = pd.DataFrame(X_doc2vec, 
                               columns=[f'doc2vec_{i}' for i in range(X_doc2vec.shape[1])])
    
    # 3. Extract sentiment features
    X_sentiment_df = pd.DataFrame({
        'body_sentiment': [TextBlob(sentence).sentiment.polarity for sentence in sentences],
        'title_sentiment': [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    })
    
    # Combine all features
    X = pd.concat([X_tfidf_df, X_doc2vec_df, X_sentiment_df], axis=1)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Convert numeric predictions to labels
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = [sentiment_labels[int(pred)] for pred in predictions]
    
    return predictions

# Main function
def main():
    # Download necessary NLTK data
    download_nltk_resources()
    
    # Load data
    file_path = 'Books_10k.jsonl'
    print("Loading data...")
    data = load_jsonl(file_path)
    
    print("Processing data and splitting reviews into sentences...")
    df = process_data_sentences(data)
    
    print(f"Total sentences extracted: {len(df)}")
    print(f"Sentiment distribution: \n{df['sentiment'].value_counts()}")
    
    # Train combined model
    print("Training combined model (CatBoost + TF-IDF + Doc2Vec)...")
    model, doc2vec_model, tfidf_vectorizer, y_true, y_pred = train_combined_model(df, iterations=100)
    
    print(f"Model accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    print(conf_matrix)
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(df, y_true, y_pred, conf_matrix)
    
    # Example: Predict sentiment for new reviews
    sample_reviews = [
        "This book was absolutely fantastic! I couldn't put it down.",
        "The characters were poorly developed and the plot was predictable.",
        "It was an okay read, nothing special but not terrible either."
    ]
    
    print("\nPredicting sentiment for sample reviews:")
    predictions = predict_with_combined_model(model, doc2vec_model, tfidf_vectorizer, sample_reviews)
    
    for i, review in enumerate(sample_reviews):
        print(f"Review: {review}")
        print(f"Predicted sentiment: {predictions[i]}")
        print("---")

if __name__ == "__main__":
    main()