import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.load_features import load_features
from src.data.create_splits import load_data_splits

os.makedirs('models', exist_ok=True)

def train_model(model_type='logreg', feature_set='all'):
    X, y = load_features(feature_set)

    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)

    try:
        train_indices, test_indices = load_data_splits(feature_set, model_type)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Split files not found for {model_type}_{feature_set}. "
            f"Please create splits first using 'python main.py --stage split --model {model_type} --features {feature_set}'"
        )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == 'logreg':
        model = LogisticRegression(
            C=0.1,
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced',
            tol=1e-4
        )
    elif model_type == 'rf':
        model = RandomForestClassifier()
    elif model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'nn':
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            alpha=0.01,
            learning_rate_init=0.001,
            early_stopping=False,
            random_state=42
        )
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)

    joblib.dump(model, f'models/{model_type}_{feature_set}.joblib')
    joblib.dump(scaler, f'models/{model_type}_{feature_set}_scaler.joblib')

    print(f"Model and scaler saved to models/{model_type}_{feature_set}.joblib and models/{model_type}_{feature_set}_scaler.joblib")
