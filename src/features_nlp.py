#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract NLP-based features from the cleaned mental health posts dataset.

This script loads the cleaned dataset and extracts the following NLP-based features:
1. TF-IDF features
2. GloVe word embedding features
3. BERT embeddings
4. (Optional) SentenceTransformers embeddings

Input:
    - data/processed/cleaned.csv: CSV file containing the cleaned dataset
      with columns: text, cleaned_text, label, timestamp

Output:
    - features/nlp/tfidf.csv: CSV file containing TF-IDF features
    - features/nlp/glove.csv: CSV file containing GloVe word embedding features
    - features/nlp/bert.csv: CSV file containing BERT embeddings
    - features/nlp/sentence_bert.csv: (Optional) CSV file containing SentenceTransformers embeddings
"""

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import nltk
import requests
import zipfile
import io

os.makedirs('features/nlp', exist_ok=True)

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_PATH = "glove.6B.100d.txt"

def download_glove():
    """Download GloVe vectors if not already downloaded."""
    if not os.path.exists(GLOVE_PATH):
        print("Downloading GloVe vectors...")
        r = requests.get(GLOVE_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extract("glove.6B.100d.txt")
        print("GloVe vectors downloaded.")

def load_glove_vectors():
    """
    Load GloVe word vectors.

    Returns:
        dict: Dictionary mapping words to their GloVe vectors
    """
    download_glove()

    print("Loading GloVe vectors...")
    word_vectors = {}
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    print(f"Loaded {len(word_vectors)} GloVe word vectors.")
    return word_vectors

def extract_tfidf_features(df=None, output_file=None, max_features=5000, input_file=None, ngram_range=(1, 1), train_indices=None, test_indices=None):
    """
    Extract TF-IDF features from the cleaned text.

    Args:
        df (pandas.DataFrame, optional): DataFrame containing the cleaned dataset
        output_file (str, optional): Path to the output CSV file
        max_features (int): Maximum number of features to extract
        input_file (str, optional): Path to the input CSV file
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted
        train_indices (numpy.ndarray, optional): Indices for training data
        test_indices (numpy.ndarray, optional): Indices for test data
    """
    print("Extracting TF-IDF features...")

    # Load data if DataFrame not provided
    if df is None and input_file is not None:
        df = pd.read_csv(input_file)

    if df is None:
        raise ValueError("Either df or input_file must be provided")

    if df.empty:
        raise ValueError("Empty input: DataFrame has no rows")

    text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    if text_column not in df.columns:
        raise ValueError(f"Text column not found. Expected 'cleaned_text' or 'text', got {df.columns.tolist()}")

    if train_indices is None or test_indices is None:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

        os.makedirs('data/splits', exist_ok=True)
        pd.DataFrame({'index': train_indices}).to_csv('data/splits/tfidf_train_indices.csv', index=False)
        pd.DataFrame({'index': test_indices}).to_csv('data/splits/tfidf_test_indices.csv', index=False)
        print("Created and saved train/test split indices.")

    train_texts = df.iloc[train_indices][text_column]
    test_texts = df.iloc[test_indices][text_column]
    train_labels = df.iloc[train_indices]['label']
    test_labels = df.iloc[test_indices]['label']

    min_df = 1 if len(train_texts) < 10 else 2

    if len(train_texts) <= 5:
        analyzer = 'char_wb'
        ngram_range = (2, 5)
    else:
        analyzer = 'word'

    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        use_idf=True,
        smooth_idf=True,
        analyzer=analyzer
    )

    tfidf_vectorizer.fit(train_texts)

    train_tfidf_matrix = tfidf_vectorizer.transform(train_texts)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_texts)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    train_tfidf_df = pd.DataFrame(train_tfidf_matrix.toarray(), columns=feature_names)
    test_tfidf_df = pd.DataFrame(test_tfidf_matrix.toarray(), columns=feature_names)

    train_tfidf_df['label'] = train_labels.values
    test_tfidf_df['label'] = test_labels.values

    tfidf_df = pd.DataFrame(index=range(len(df)), columns=feature_names + ['label'])
    tfidf_df.iloc[train_indices] = train_tfidf_df
    tfidf_df.iloc[test_indices] = test_tfidf_df

    if output_file is not None:
        tfidf_df.to_csv(output_file, index=False)
        print(f"TF-IDF features saved to {output_file}")

    print(f"TF-IDF features shape: {tfidf_df.shape}")
    return tfidf_df

def extract_glove_features(df, output_file=None, word_vectors=None, train_indices=None, test_indices=None):
    """
    Extract GloVe word embedding features from the cleaned text.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset
        output_file (str, optional): Path to the output CSV file
        word_vectors (dict, optional): Dictionary mapping words to their GloVe vectors
        train_indices (numpy.ndarray, optional): Indices for training data
        test_indices (numpy.ndarray, optional): Indices for test data

    Returns:
        pandas.DataFrame: DataFrame containing GloVe features
    """
    print("Extracting GloVe word embedding features...")

    if word_vectors is None:
        word_vectors = load_glove_vectors()

    text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    if text_column not in df.columns:
        raise ValueError(f"Text column not found. Expected 'cleaned_text' or 'text', got {df.columns.tolist()}")

    if train_indices is None or test_indices is None:
        try:
            train_indices = pd.read_csv('data/splits/tfidf_train_indices.csv')['index'].values
            test_indices = pd.read_csv('data/splits/tfidf_test_indices.csv')['index'].values
            print("Using train/test split indices from TF-IDF feature extraction.")
        except FileNotFoundError:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(df))
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            os.makedirs('data/splits', exist_ok=True)
            pd.DataFrame({'index': train_indices}).to_csv('data/splits/glove_train_indices.csv', index=False)
            pd.DataFrame({'index': test_indices}).to_csv('data/splits/glove_test_indices.csv', index=False)
            print("Created and saved train/test split indices.")

    vector_dim = next(iter(word_vectors.values())).shape[0]

    glove_features = np.zeros((len(df), vector_dim))

    for i, text in enumerate(df[text_column]):
        words = text.split()
        word_vectors_sum = np.zeros(vector_dim)
        count = 0

        for word in words:
            if word in word_vectors:
                word_vectors_sum += word_vectors[word]
                count += 1

        if count > 0:
            glove_features[i] = word_vectors_sum / count

    columns = [f'glove_{i}' for i in range(vector_dim)]
    glove_df = pd.DataFrame(glove_features, columns=columns)

    glove_df['label'] = df['label'].values

    if output_file is not None:
        glove_df.to_csv(output_file, index=False)
        print(f"GloVe word embedding features saved to {output_file}")

    print(f"GloVe features shape: {glove_df.shape}")
    return glove_df

def extract_bert_features(df, output_file, model_name='distilbert-base-uncased', train_indices=None, test_indices=None):
    """
    Extract BERT embeddings from the cleaned text.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset
        output_file (str): Path to the output CSV file
        model_name (str): Name of the BERT model to use
        train_indices (numpy.ndarray, optional): Indices for training data
        test_indices (numpy.ndarray, optional): Indices for test data
    """
    print(f"Extracting BERT embeddings using {model_name}...")

    if train_indices is None or test_indices is None:
        try:
            train_indices = pd.read_csv('data/splits/tfidf_train_indices.csv')['index'].values
            test_indices = pd.read_csv('data/splits/tfidf_test_indices.csv')['index'].values
            print("Using train/test split indices from TF-IDF feature extraction.")
        except FileNotFoundError:
            try:
                train_indices = pd.read_csv('data/splits/glove_train_indices.csv')['index'].values
                test_indices = pd.read_csv('data/splits/glove_test_indices.csv')['index'].values
                print("Using train/test split indices from GloVe feature extraction.")
            except FileNotFoundError:
                # If not found, create a new split
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(df))
                train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

                # Save the indices for later use
                os.makedirs('data/splits', exist_ok=True)
                pd.DataFrame({'index': train_indices}).to_csv('data/splits/bert_train_indices.csv', index=False)
                pd.DataFrame({'index': test_indices}).to_csv('data/splits/bert_test_indices.csv', index=False)
                print("Created and saved train/test split indices.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    bert_features = np.zeros((len(df), model.config.hidden_size))

    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch_texts = df['cleaned_text'].iloc[i:i+batch_size].tolist()

        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        end_idx = min(i + batch_size, len(df))
        bert_features[i:end_idx] = embeddings

    columns = [f'bert_{i}' for i in range(bert_features.shape[1])]
    bert_df = pd.DataFrame(bert_features, columns=columns)

    bert_df['label'] = df['label'].values

    bert_df.to_csv(output_file, index=False)
    print(f"BERT embeddings saved to {output_file}")
    print(f"BERT features shape: {bert_df.shape}")

    return bert_df

def extract_sentence_bert_features(df, output_file, model_name='all-MiniLM-L6-v2', train_indices=None, test_indices=None):
    """
    Extract SentenceTransformers embeddings from the cleaned text.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset
        output_file (str): Path to the output CSV file
        model_name (str): Name of the SentenceTransformers model to use
        train_indices (numpy.ndarray, optional): Indices for training data
        test_indices (numpy.ndarray, optional): Indices for test data
    """
    print(f"Extracting SentenceTransformers embeddings using {model_name}...")

    if train_indices is None or test_indices is None:
        try:
            train_indices = pd.read_csv('data/splits/tfidf_train_indices.csv')['index'].values
            test_indices = pd.read_csv('data/splits/tfidf_test_indices.csv')['index'].values
            print("Using train/test split indices from TF-IDF feature extraction.")
        except FileNotFoundError:
            try:
                train_indices = pd.read_csv('data/splits/glove_train_indices.csv')['index'].values
                test_indices = pd.read_csv('data/splits/glove_test_indices.csv')['index'].values
                print("Using train/test split indices from GloVe feature extraction.")
            except FileNotFoundError:
                try:
                    train_indices = pd.read_csv('data/splits/bert_train_indices.csv')['index'].values
                    test_indices = pd.read_csv('data/splits/bert_test_indices.csv')['index'].values
                    print("Using train/test split indices from BERT feature extraction.")
                except FileNotFoundError:
                    from sklearn.model_selection import train_test_split
                    indices = np.arange(len(df))
                    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

                    os.makedirs('data/splits', exist_ok=True)
                    pd.DataFrame({'index': train_indices}).to_csv('data/splits/sbert_train_indices.csv', index=False)
                    pd.DataFrame({'index': test_indices}).to_csv('data/splits/sbert_test_indices.csv', index=False)
                    print("Created and saved train/test split indices.")

    model = SentenceTransformer(model_name)

    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)

    columns = [f'sbert_{i}' for i in range(embeddings.shape[1])]
    sbert_df = pd.DataFrame(embeddings, columns=columns)

    sbert_df['label'] = df['label'].values

    sbert_df.to_csv(output_file, index=False)
    print(f"SentenceTransformers embeddings saved to {output_file}")
    print(f"SentenceTransformers features shape: {sbert_df.shape}")

    return sbert_df

def main():
    """Main function to extract NLP-based features."""
    parser = argparse.ArgumentParser(description='Extract NLP-based features from mental health posts dataset')
    parser.add_argument('--input', type=str, default='data/processed/cleaned.csv',
                        help='Input file path (default: data/processed/cleaned.csv)')
    parser.add_argument('--tfidf_output', type=str, default='features/nlp/tfidf.csv',
                        help='Output file path for TF-IDF features (default: features/nlp/tfidf.csv)')
    parser.add_argument('--glove_output', type=str, default='features/nlp/glove.csv',
                        help='Output file path for GloVe features (default: features/nlp/glove.csv)')
    parser.add_argument('--bert_output', type=str, default='features/nlp/bert.csv',
                        help='Output file path for BERT features (default: features/nlp/bert.csv)')
    parser.add_argument('--sbert_output', type=str, default='features/nlp/sentence_bert.csv',
                        help='Output file path for SentenceTransformers features (default: features/nlp/sentence_bert.csv)')
    parser.add_argument('--skip_sbert', action='store_true',
                        help='Skip SentenceTransformers feature extraction')
    parser.add_argument('--use_existing_split', action='store_true',
                        help='Use existing train/test split indices if available')
    args = parser.parse_args()

    print(f"Loading cleaned dataset from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples.")

    train_indices = None
    test_indices = None

    if args.use_existing_split:
        try:
            train_indices = pd.read_csv('data/splits/logreg_all_train_indices.csv')['index'].values
            test_indices = pd.read_csv('data/splits/logreg_all_test_indices.csv')['index'].values
            print("Using train/test split indices from model training.")
        except FileNotFoundError:
            print("No existing train/test split indices found from model training.")

    tfidf_result = extract_tfidf_features(df, args.tfidf_output, train_indices=train_indices, test_indices=test_indices)

    if train_indices is None or test_indices is None:
        try:
            train_indices = pd.read_csv('data/splits/tfidf_train_indices.csv')['index'].values
            test_indices = pd.read_csv('data/splits/tfidf_test_indices.csv')['index'].values
            print("Using train/test split indices from TF-IDF feature extraction for subsequent features.")
        except FileNotFoundError:
            print("Warning: Could not load train/test split indices from TF-IDF feature extraction.")

    word_vectors = load_glove_vectors()
    extract_glove_features(df, args.glove_output, word_vectors, train_indices=train_indices, test_indices=test_indices)

    extract_bert_features(df, args.bert_output, train_indices=train_indices, test_indices=test_indices)

    if not args.skip_sbert:
        extract_sentence_bert_features(df, args.sbert_output, train_indices=train_indices, test_indices=test_indices)

    print("NLP-based feature extraction completed.")

if __name__ == "__main__":
    main()
