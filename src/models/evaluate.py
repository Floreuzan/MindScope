import joblib
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.utils.load_features import load_features
from src.data.create_splits import load_data_splits

os.makedirs('results', exist_ok=True)

def safe_predict(model, X):
    """
    Make predictions with a model while suppressing RuntimeWarnings.

    Args:
        model: The trained model
        X: The feature matrix

    Returns:
        Predictions from the model
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return model.predict(X)

def safe_predict_proba(model, X):
    """
    Make probability predictions with a model while suppressing RuntimeWarnings.

    Args:
        model: The trained model
        X: The feature matrix

    Returns:
        Probability predictions from the model
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return model.predict_proba(X)

def evaluate_model(model_type='logreg', feature_set='all'):
    model = joblib.load(f'models/{model_type}_{feature_set}.joblib')
    X, y = load_features(feature_set)

    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)

    try:
        scaler = joblib.load(f'models/{model_type}_{feature_set}_scaler.joblib')
        X = scaler.transform(X)
    except FileNotFoundError:
        print(f"Warning: Scaler not found for {model_type}_{feature_set}. Using unscaled features.")

    if hasattr(X, 'values'):
        X = X.values

    try:
        _, test_indices = load_data_splits(feature_set, model_type)

        X_test = X[test_indices]
        y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]

        print(f"Using saved test indices from data/splits/{model_type}_{feature_set}_test_indices.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Split files not found for {model_type}_{feature_set}. "
            f"Please create splits first using 'python main.py --stage split --model {model_type} --features {feature_set}'"
        )

    y_pred = safe_predict(model, X_test)
    y_proba = safe_predict_proba(model, X_test)[:, 1]

    y = y_test

    classification_rep = classification_report(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    conf_matrix = confusion_matrix(y, y_pred)

    print(classification_rep)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:\n", conf_matrix)

    with open('results/evaluation_metrics.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_rep)
        f.write("\nROC AUC: {}\n".format(roc_auc))
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

    print(f"Evaluation metrics saved to results/evaluation_metrics.txt")
