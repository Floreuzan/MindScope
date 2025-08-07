import pandas as pd
import numpy as np
import warnings

def load_features(feature_set='all'):
    """
    Load features from CSV files based on the specified feature set.

    Args:
        feature_set (str): The feature set to load. Options are 'structured', 
                          'tfidf', 'glove', 'bert', or 'all'.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    if feature_set == 'structured':
        df = pd.read_csv('features/structured/structured_features.csv')
        y = df['label']
        X = df.drop('label', axis=1)
        return X, y

    elif feature_set == 'tfidf':
        df = pd.read_csv('features/nlp/tfidf.csv')
        y = df['label']
        X = df.drop('label', axis=1)
        return X, y

    elif feature_set == 'glove':
        df = pd.read_csv('features/nlp/glove.csv')
        y = df['label']
        X = df.drop('label', axis=1)
        return X, y

    elif feature_set == 'bert':
        df = pd.read_csv('features/nlp/bert.csv')
        y = df['label']
        X = df.drop('label', axis=1)
        return X, y

    elif feature_set == 'all':
        structured = pd.read_csv('features/structured/structured_features.csv')
        tfidf = pd.read_csv('features/nlp/tfidf.csv')
        glove = pd.read_csv('features/nlp/glove.csv')
        bert = pd.read_csv('features/nlp/bert.csv')

        row_counts = [len(structured), len(tfidf), len(glove), len(bert)]
        if len(set(row_counts)) > 1:
            warnings.warn(f"Feature files have different numbers of rows: {row_counts}. This may lead to incorrect results.")

        has_label = {
            'structured': 'label' in structured.columns,
            'tfidf': 'label' in tfidf.columns,
            'glove': 'label' in glove.columns,
            'bert': 'label' in bert.columns
        }

        if not all(has_label.values()):
            missing_labels = [name for name, has in has_label.items() if not has]
            warnings.warn(f"The following feature files don't have a 'label' column: {missing_labels}. Using labels from structured features.")
            labels_identical = False
        else:
            labels_identical = (np.array_equal(structured['label'].values, tfidf['label'].values) and
                               np.array_equal(structured['label'].values, glove['label'].values) and
                               np.array_equal(structured['label'].values, bert['label'].values))

            if not labels_identical:
                warnings.warn("Label columns in feature files are not identical. Using labels from structured features.")

        y = structured['label']

        structured = structured.drop('label', axis=1)

        feature_dfs = [structured]


        for feature_df, name in [(tfidf, 'tfidf'), (glove, 'glove'), (bert, 'bert')]:
            if 'label' in feature_df.columns:
                feature_cols = feature_df.drop('label', axis=1)
            else:
                feature_cols = feature_df
            feature_dfs.append(feature_cols)

        df = pd.concat(feature_dfs, axis=1)
    else:
        raise ValueError(f"Invalid feature set: {feature_set}. Valid options are 'structured', 'tfidf', 'glove', 'bert', or 'all'.")

    return df, y
