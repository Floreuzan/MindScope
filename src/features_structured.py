#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract structured-based features from the cleaned mental health posts dataset.

This script loads the cleaned dataset and extracts the following structured-based features:
1. Post length features (character length, word count)
2. Sentiment analysis features (TextBlob polarity, VADER sentiment)
3. Temporal features (hour of day, day of week)
4. Stylometric features (average sentence length, pronoun percentage, negation word count, emotional tone)

Input:
    - data/processed/cleaned.csv: CSV file containing the cleaned dataset
      with columns: text, cleaned_text, label, timestamp

Output:
    - features/structured/structured_features.csv: CSV file containing structured features
"""

import os
import pandas as pd
import numpy as np
import argparse
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime
import re

nltk.download('vader_lexicon', quiet=True)

os.makedirs('features/structured', exist_ok=True)

def extract_post_length_features(df):
    """
    Extract post length features from the text.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset

    Returns:
        pandas.DataFrame: DataFrame containing post length features
    """
    print("Extracting post length features...")

    df['char_length'] = df['cleaned_text'].str.len()

    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    return df[['char_length', 'word_count']]

def extract_sentiment_features(df):
    """
    Extract sentiment analysis features from the text.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset

    Returns:
        pandas.DataFrame: DataFrame containing sentiment analysis features
    """
    print("Extracting sentiment analysis features...")

    df['textblob_sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    sid = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_pos'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['pos'])
    df['vader_neg'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['neg'])
    df['vader_neu'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['neu'])

    return df[['textblob_sentiment', 'vader_sentiment', 'vader_pos', 'vader_neg', 'vader_neu']]

def extract_temporal_features(df):
    """
    Extract temporal features from the timestamp.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset

    Returns:
        pandas.DataFrame: DataFrame containing temporal features
    """
    print("Extracting temporal features...")

    df['datetime'] = pd.to_datetime(df['timestamp'])

    df['hour_of_day'] = df['datetime'].dt.hour

    df['day_of_week'] = df['datetime'].dt.dayofweek

    df['month'] = df['datetime'].dt.month

    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 0  # morning
        elif 12 <= hour < 17:
            return 1  # afternoon
        elif 17 <= hour < 22:
            return 2  # evening
        else:
            return 3  # night

    df['time_of_day'] = df['hour_of_day'].apply(get_time_of_day)

    return df[['hour_of_day', 'day_of_week', 'month', 'is_weekend', 'time_of_day']]

def extract_stylometric_features(df):
    """
    Extract stylometric features from the text.

    Args:
        df (pandas.DataFrame): DataFrame containing the cleaned dataset

    Returns:
        pandas.DataFrame: DataFrame containing stylometric features
    """
    print("Extracting stylometric features...")

    if 'word_count' not in df.columns:
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    def avg_sentence_length(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        return np.mean([len(s.split()) for s in sentences])

    df['avg_sentence_length'] = df['cleaned_text'].apply(avg_sentence_length)

    def pronoun_percentage(text):
        words = text.lower().split()
        if not words:
            return 0
        pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        pronoun_count = sum(1 for word in words if word in pronouns)
        return pronoun_count / len(words) * 100

    df['pronoun_percentage'] = df['cleaned_text'].apply(pronoun_percentage)

    def first_person_singular_percentage(text):
        words = text.lower().split()
        if not words:
            return 0
        pronouns = ['i', 'me', 'my', 'mine', 'myself']
        pronoun_count = sum(1 for word in words if word in pronouns)
        return pronoun_count / len(words) * 100

    df['first_person_singular_percentage'] = df['cleaned_text'].apply(first_person_singular_percentage)

    def negation_word_count(text):
        words = text.lower().split()
        negations = ['not', 'never', 'no', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere', 'cannot', "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't"]
        return sum(1 for word in words if word in negations)

    df['negation_word_count'] = df['cleaned_text'].apply(negation_word_count)

    def emotional_tone(text):
        positive_words = ['happy', 'joy', 'love', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'awesome', 'fantastic', 'positive', 'hope', 'hopeful', 'excited', 'exciting', 'pleased', 'glad', 'satisfied', 'content', 'grateful', 'thankful', 'appreciate', 'appreciated', 'proud', 'confident', 'optimistic', 'calm', 'peaceful', 'relaxed', 'relieved']
        negative_words = ['sad', 'unhappy', 'depressed', 'depression', 'anxious', 'anxiety', 'worry', 'worried', 'fear', 'scared', 'afraid', 'terrified', 'panic', 'stressed', 'stress', 'angry', 'mad', 'upset', 'frustrated', 'annoyed', 'irritated', 'hate', 'dislike', 'disappointed', 'disappointing', 'hopeless', 'helpless', 'alone', 'lonely', 'isolated', 'abandoned', 'rejected', 'hurt', 'pain', 'suffering', 'miserable', 'worthless', 'useless', 'guilty', 'ashamed', 'embarrassed', 'disgusted', 'tired', 'exhausted', 'overwhelmed', 'suicidal', 'suicide', 'die', 'death', 'kill', 'end', 'empty', 'numb', 'lost', 'confused', 'broken', 'damaged', 'ruined', 'failure', 'failed', 'mistake', 'regret']

        words = text.lower().split()
        if not words:
            return 0, 0

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        return positive_count, negative_count

    df['positive_word_count'], df['negative_word_count'] = zip(*df['cleaned_text'].apply(emotional_tone))

    df['positive_word_percentage'] = df.apply(
        lambda x: (x['positive_word_count'] / x['word_count'] * 100) if x['word_count'] > 0 else 0, 
        axis=1
    )
    df['negative_word_percentage'] = df.apply(
        lambda x: (x['negative_word_count'] / x['word_count'] * 100) if x['word_count'] > 0 else 0, 
        axis=1
    )

    df['emotional_tone_ratio'] = df.apply(
        lambda x: x['positive_word_count'] / x['negative_word_count'] if x['negative_word_count'] > 0 else 
                 (1.0 if x['positive_word_count'] == 0 else 1000.0), 
        axis=1
    )

    return df[['avg_sentence_length', 'pronoun_percentage', 'first_person_singular_percentage', 'negation_word_count', 'positive_word_count', 'negative_word_count', 'positive_word_percentage', 'negative_word_percentage', 'emotional_tone_ratio']]

def extract_structured_features(input_file=None, output_file=None, df=None):
    """
    Extract structured features from the cleaned dataset.

    Args:
        input_file (str, optional): Path to the input CSV file
        output_file (str, optional): Path to the output CSV file
        df (pandas.DataFrame, optional): DataFrame containing the cleaned dataset

    Returns:
        pandas.DataFrame: DataFrame containing structured features
    """
    if df is None and input_file is not None:
        print(f"Loading cleaned dataset from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples.")
    elif df is not None:
        print(f"Using provided DataFrame with {len(df)} samples.")
    else:
        raise ValueError("Either df or input_file must be provided")

    post_length_features = extract_post_length_features(df)
    sentiment_features = extract_sentiment_features(df)
    temporal_features = extract_temporal_features(df)
    stylometric_features = extract_stylometric_features(df)

    features_df = pd.concat([
        post_length_features,
        sentiment_features,
        temporal_features,
        stylometric_features,
        df[['label']]
    ], axis=1)

    if output_file is not None:
        features_df.to_csv(output_file, index=False)
        print(f"Structured features saved to {output_file}")

    print(f"Structured features shape: {features_df.shape}")

    print("\nFeature statistics:")
    print(features_df.describe().transpose())

    return features_df

def main():
    """Main function to extract structured features."""
    parser = argparse.ArgumentParser(description='Extract structured features from mental health posts dataset')
    parser.add_argument('--input', type=str, default='data/processed/cleaned.csv',
                        help='Input file path (default: data/processed/cleaned.csv)')
    parser.add_argument('--output', type=str, default='features/structured/structured_features.csv',
                        help='Output file path (default: features/structured/structured_features.csv)')
    args = parser.parse_args()

    extract_structured_features(args.input, args.output)
    print("Structured feature extraction completed.")

if __name__ == "__main__":
    main()
