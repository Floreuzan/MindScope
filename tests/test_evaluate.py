import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import evaluate_model, safe_predict, safe_predict_proba

class TestEvaluateModel:
    """Test suite for model evaluation functionality."""

    @pytest.fixture
    def mock_features(self):
        """Fixture to create mock features for testing."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([0, 1, 0, 1])
        return X, y

    @pytest.fixture
    def mock_model(self):
        """Fixture to create a mock trained model."""
        model = LogisticRegression()
        model.predict = MagicMock(return_value=np.array([0, 1, 0, 1]))
        model.predict_proba = MagicMock(return_value=np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]))
        return model

    @pytest.fixture
    def mock_scaler(self):
        """Fixture to create a mock scaler."""
        scaler = StandardScaler()
        scaler.transform = MagicMock(return_value=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
        return scaler

    @pytest.fixture
    def mock_joblib_load(self, mock_model, mock_scaler):
        """Fixture to mock joblib.load function."""
        def side_effect(filename):
            if 'scaler' in filename:
                return mock_scaler
            return mock_model

        with patch('src.models.evaluate.joblib.load') as mock:
            mock.side_effect = side_effect
            yield mock

    @pytest.fixture
    def mock_load_features(self, mock_features):
        """Fixture to mock the load_features function."""
        with patch('src.models.evaluate.load_features') as mock:
            mock.return_value = mock_features
            yield mock

    @pytest.fixture
    def mock_makedirs(self):
        """Fixture to mock os.makedirs function."""
        with patch('src.models.evaluate.os.makedirs') as mock:
            yield mock

    @pytest.fixture
    def mock_open_file(self):
        """Fixture to mock open function for file writing."""
        with patch('builtins.open', mock_open()) as mock:
            yield mock

    def test_safe_predict(self, mock_model):
        """Test that safe_predict suppresses RuntimeWarnings."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_model.predict.return_value = np.array([0, 1])

        result = safe_predict(mock_model, X)

        mock_model.predict.assert_called_once_with(X)

        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_safe_predict_proba(self, mock_model):
        """Test that safe_predict_proba suppresses RuntimeWarnings."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

        result = safe_predict_proba(mock_model, X)

        mock_model.predict_proba.assert_called_once_with(X)

        expected = np.array([[0.9, 0.1], [0.2, 0.8]])
        np.testing.assert_array_equal(result, expected)

    def test_evaluate_model_basic(self, mock_joblib_load, mock_load_features, mock_makedirs, mock_open_file):
        """Test basic functionality of evaluate_model."""
        X, y = mock_load_features.return_value

        test_indices = np.array([2, 3])

        with patch('src.models.evaluate.load_data_splits', return_value=(None, test_indices)), \
             patch('src.models.evaluate.safe_predict', return_value=np.array([0, 1])), \
             patch('src.models.evaluate.safe_predict_proba', return_value=np.array([[0.9, 0.1], [0.2, 0.8]])):
            evaluate_model(model_type='logreg', feature_set='structured')

            assert mock_joblib_load.call_count == 2
            mock_joblib_load.assert_any_call('models/logreg_structured.joblib')
            mock_joblib_load.assert_any_call('models/logreg_structured_scaler.joblib')

            mock_load_features.assert_called_once_with('structured')

            mock_open_file.assert_called_once_with('results/evaluation_metrics.txt', 'w')

            handle = mock_open_file()
            assert handle.write.call_count > 0

    def test_evaluate_model_missing_scaler(self, mock_load_features, mock_makedirs, mock_open_file):
        """Test that evaluate_model handles missing scaler files gracefully."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

        X, y = mock_load_features.return_value

        test_indices = np.array([2, 3])

        def mock_load_side_effect(filename):
            if 'scaler' in filename:
                raise FileNotFoundError("Scaler not found")
            return mock_model

        with patch('src.models.evaluate.joblib.load', side_effect=mock_load_side_effect), \
             patch('src.models.evaluate.load_data_splits', return_value=(None, test_indices)), \
             patch('src.models.evaluate.safe_predict', return_value=np.array([0, 1])), \
             patch('src.models.evaluate.safe_predict_proba', return_value=np.array([[0.9, 0.1], [0.2, 0.8]])):
            evaluate_model(model_type='logreg', feature_set='structured')

            mock_open_file.assert_called_once_with('results/evaluation_metrics.txt', 'w')

    def test_evaluate_model_handles_nan_inf(self, mock_joblib_load, mock_makedirs, mock_open_file):
        """Test that evaluate_model handles NaN and infinite values."""
        X = pd.DataFrame({
            'feature1': [1.0, np.nan, np.inf, -np.inf],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([0, 1, 0, 1])

        test_indices = np.array([2, 3])

        mock_model = mock_joblib_load.return_value

        with patch('src.models.evaluate.load_features', return_value=(X, y)), \
             patch('src.models.evaluate.load_data_splits', return_value=(None, test_indices)), \
             patch('src.models.evaluate.safe_predict', return_value=np.array([0, 1])), \
             patch('src.models.evaluate.safe_predict_proba', return_value=np.array([[0.9, 0.1], [0.2, 0.8]])):
            evaluate_model(model_type='logreg', feature_set='structured')

            mock_open_file.assert_called_once_with('results/evaluation_metrics.txt', 'w')

    def test_evaluate_model_split_not_found(self, mock_joblib_load, mock_load_features, mock_makedirs, mock_open_file):
        """Test that evaluate_model raises FileNotFoundError when split files don't exist."""
        with patch('src.models.evaluate.load_data_splits', side_effect=FileNotFoundError("Split files not found")):
            with pytest.raises(FileNotFoundError, match="Split files not found"):
                evaluate_model(model_type='logreg', feature_set='structured')
