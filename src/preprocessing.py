#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess the generated mental health posts dataset.

This script loads the raw dataset, cleans the text by removing mentions,
emojis, links, hashtags, punctuation, and performs lowercase conversion,
tokenization, lemmatization, and stopword removal.

Input:
    - data/raw/generated_dataset.csv: CSV file containing the generated dataset
      with columns: text, label, timestamp

Output:
    - data/processed/cleaned.csv: CSV file containing the cleaned dataset
      with columns: text, cleaned_text, label, timestamp
"""

import os
import re
import pandas as pd
import nltk
import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

os.makedirs('data/processed', exist_ok=True)

def clean_text(text):
    """
    Clean text by removing mentions, emojis, links, hashtags, punctuation,
    and performing lowercase conversion, tokenization, lemmatization, and stopword removal.

    Args:
        text (str): The text to clean

    Returns:
        str: The cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove hashtags (#topic)
    text = re.sub(r'#\w+', '', text)

    # Remove emojis (simple approach)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    cleaned_text = ' '.join(tokens)

    return cleaned_text

def preprocess_dataset(input_file, output_file):
    """
    Preprocess the dataset by cleaning the text.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    df = pd.read_csv(input_file)

    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")

    print(f"Dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Average original text length: {df['text'].str.len().mean():.1f} characters")
    print(f"  Average cleaned text length: {df['cleaned_text'].str.len().mean():.1f} characters")
    print(f"  Reduction in text length: {(1 - df['cleaned_text'].str.len().mean() / df['text'].str.len().mean()) * 100:.1f}%")

def main():
    """Main function to preprocess the dataset."""
    parser = argparse.ArgumentParser(description='Preprocess mental health posts dataset')
    parser.add_argument('--input', type=str, default='data/raw/generated_dataset.csv',
                        help='Input file path (default: data/raw/generated_dataset.csv)')
    parser.add_argument('--output', type=str, default='data/processed/cleaned.csv',
                        help='Output file path (default: data/processed/cleaned.csv)')
    args = parser.parse_args()

    print(f"Preprocessing dataset from {args.input}...")
    preprocess_dataset(args.input, args.output)

if __name__ == "__main__":
    main()