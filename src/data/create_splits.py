import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.load_features import load_features

def create_data_splits(feature_set='all', model_type='logreg', test_size=0.2, random_state=42):
    """
    Create and save train/test splits for a given feature set and model type.

    Args:
        feature_set (str): The feature set to use. Options are 'structured', 
                          'tfidf', 'glove', 'bert', or 'all'.
        model_type (str): The model type to use. Options are 'logreg', 'rf', 'svm', 'nn'.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_indices, test_indices) - The indices for train and test sets
    """
    os.makedirs('data/splits', exist_ok=True)

    X, y = load_features(feature_set)

    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    pd.DataFrame({'index': train_indices}).to_csv(
        f'data/splits/{model_type}_{feature_set}_train_indices.csv', index=False
    )
    pd.DataFrame({'index': test_indices}).to_csv(
        f'data/splits/{model_type}_{feature_set}_test_indices.csv', index=False
    )

    print(f"Train/test splits saved to data/splits/{model_type}_{feature_set}_train_indices.csv and data/splits/{model_type}_{feature_set}_test_indices.csv")

    return train_indices, test_indices

def load_data_splits(feature_set='all', model_type='logreg'):
    """
    Load previously created train/test splits for a given feature set and model type.

    Args:
        feature_set (str): The feature set to use. Options are 'structured', 
                          'tfidf', 'glove', 'bert', or 'all'.
        model_type (str): The model type to use. Options are 'logreg', 'rf', 'svm', 'nn'.

    Returns:
        tuple: (train_indices, test_indices) - The indices for train and test sets

    Raises:
        FileNotFoundError: If the split files don't exist
    """
    try:
        train_indices = pd.read_csv(f'data/splits/{model_type}_{feature_set}_train_indices.csv')['index'].values
        test_indices = pd.read_csv(f'data/splits/{model_type}_{feature_set}_test_indices.csv')['index'].values

        return train_indices, test_indices
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Split files not found for {model_type}_{feature_set}. "
            f"Please create splits first using 'python main.py --stage split --model {model_type} --features {feature_set}'"
        )
