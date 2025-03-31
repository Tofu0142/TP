import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import numpy as np
from textblob import TextBlob

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

# Process the data into a DataFrame
def process_data(data):
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Extract review length
    df['review_length'] = df['text'].apply(lambda x: len(x))
    
    # Extract sentiment using TextBlob
    df['body_sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    return df

# Analyze common words in negative reviews
def analyze_common_words(df, min_rating=2.0):
    negative_reviews = df[df['rating'] <= min_rating]['text']
    
    # Combine all text
    all_text = ' '.join(negative_reviews)
    
    # Remove special characters and convert to lowercase
    all_text = re.sub(r'[^\w\s]', '', all_text.lower())
    
    # Remove common stop words
    stop_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'was', 'for', 
                  'this', 'with', 'on', 'as', 'are', 'at', 'be', 'by', 'have', 'had', 
                  'has', 'i', 'you', 'my', 'book', 'not', 'but']
    
    words = [word for word in all_text.split() if word not in stop_words and len(word) > 2]
    
    return Counter(words).most_common(30)

# Generate visualizations
def generate_visualizations(df):
    # Set up the plotting environment
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Rating distribution
    ax1 = fig.add_subplot(2, 2, 1)
    sns.countplot(x='rating', data=df, palette='viridis', ax=ax1)
    ax1.set_title('Distribution of Ratings', fontsize=16)
    ax1.set_xlabel('Rating', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    
    # 2. Review length vs. rating
    ax2 = fig.add_subplot(2, 2, 2)
    sns.boxplot(x='rating', y='review_length', hue='rating', data=df, palette='viridis', ax=ax2, legend=False)
    ax2.set_title('Review Length by Rating', fontsize=16)
    ax2.set_xlabel('Rating', fontsize=12)
    ax2.set_ylabel('Review Length (characters)', fontsize=12)
    
    # 3. Sentiment distribution
    ax3 = fig.add_subplot(2, 2, 3)
    sns.histplot(df['sentiment'], bins=30, kde=True, ax=ax3, color='purple')
    ax3.set_title('Sentiment Distribution', fontsize=16)
    ax3.set_xlabel('Sentiment Score', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    
    # 4. Word cloud of negative reviews
    ax4 = fig.add_subplot(2, 2, 4)
    negative_text = ' '.join(df[df['rating'] <= 2.0]['text'])
    wordcloud = WordCloud(width=800, height=600, background_color='white', 
                         max_words=100, contour_width=3, contour_color='steelblue').generate(negative_text)
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.set_title('Common Words in Negative Reviews', fontsize=16)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('book_review_analysis.png')
    plt.close()

# Analyze verified vs. non-verified purchases
def analyze_verified_purchases(df):
    verified_stats = df.groupby('verified_purchase').agg({
        'rating': 'mean',
        'review_length': 'mean',
        'helpful_vote': 'mean',
        'sentiment': 'mean'
    }).reset_index()
    
    return verified_stats

# Main function
def main():
    file_path = 'Books_10k.jsonl'
    
    print("Loading data...")
    data = load_jsonl(file_path)
    
    print("Processing data...")
    df = process_data(data)
    
    print("Basic statistics:")
    print(f"Total reviews: {len(df)}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print(f"Average review length: {df['review_length'].mean():.2f} characters")
    
    print("\nMost common words in negative reviews:")
    common_words = analyze_common_words(df)
    for word, count in common_words:
        print(f"{word}: {count}")
    
    print("\nVerified vs. Non-verified purchase statistics:")
    verified_stats = analyze_verified_purchases(df)
    print(verified_stats)
    
    print("\nGenerating visualizations...")
    generate_visualizations(df)
    
    print("Analysis complete! Check 'book_review_analysis.png' for visualizations.")


if __name__ == "__main__":
    main()