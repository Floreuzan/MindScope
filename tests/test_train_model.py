import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.models.train_model import train_model

class TestTrainModel:
    """Test suite for model training functionality."""

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
    def mock_load_features(self, mock_features):
        """Fixture to mock the load_features function."""
        with patch('src.models.train_model.load_features') as mock:
            mock.return_value = mock_features
            yield mock

    @pytest.fixture
    def mock_joblib_dump(self):
        """Fixture to mock joblib.dump function."""
        with patch('src.models.train_model.joblib.dump') as mock:
            yield mock


    def test_train_model_logreg(self, mock_load_features, mock_joblib_dump):
        """Test training a logistic regression model."""
        with patch('src.models.train_model.load_data_splits') as mock_load_splits:
            train_indices = np.array([0, 1, 2])
            test_indices = np.array([3])
            mock_load_splits.return_value = (train_indices, test_indices)

            train_model(model_type='logreg', feature_set='structured')

        mock_load_features.assert_called_once_with('structured')

        assert mock_joblib_dump.call_count == 2

        args, _ = mock_joblib_dump.call_args_list[0]
        assert isinstance(args[0], LogisticRegression)
        assert args[1] == 'models/logreg_structured.joblib'

        args, _ = mock_joblib_dump.call_args_list[1]
        assert isinstance(args[0], StandardScaler)
        assert args[1] == 'models/logreg_structured_scaler.joblib'

    @pytest.mark.parametrize("model_type,model_class", [
        ('rf', RandomForestClassifier),
        ('svm', SVC),
        ('nn', MLPClassifier)
    ])
    def test_train_model_other_types(self, mock_load_features, mock_joblib_dump, model_type, model_class):
        """Test training other model types."""
        with patch('src.models.train_model.load_data_splits') as mock_load_splits:
            train_indices = np.array([0, 1, 2])
            test_indices = np.array([3])
            mock_load_splits.return_value = (train_indices, test_indices)

            train_model(model_type=model_type, feature_set='structured')

        args, _ = mock_joblib_dump.call_args_list[0]
        assert isinstance(args[0], model_class)
        assert args[1] == f'models/{model_type}_structured.joblib'

    def test_train_model_invalid_type(self, mock_load_features):
        """Test that an invalid model type raises a ValueError."""
        with patch('src.models.train_model.load_data_splits') as mock_load_splits:
            train_indices = np.array([0, 1, 2])
            test_indices = np.array([3])
            mock_load_splits.return_value = (train_indices, test_indices)

            with pytest.raises(ValueError, match="Unsupported model type"):
                train_model(model_type='invalid_type', feature_set='structured')

    def test_train_model_handles_nan_inf(self, mock_joblib_dump):
        """Test that the function handles NaN and infinite values."""
        X = pd.DataFrame({
            'feature1': [1.0, np.nan, np.inf, -np.inf],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        })
        y = pd.Series([0, 1, 0, 1])

        with patch('src.models.train_model.load_features', return_value=(X, y)):
            with patch('src.models.train_model.load_data_splits') as mock_load_splits:
                mock_load_splits.return_value = (np.array([0, 1]), np.array([2, 3]))

                train_model(model_type='logreg', feature_set='structured')

                assert mock_joblib_dump.call_count == 2

    def test_train_model_with_train_test_split(self, mock_load_features, mock_joblib_dump):
        """Test that the function uses the correct train/test split."""
        with patch('src.models.train_model.load_data_splits') as mock_load_splits:
            train_indices = np.array([0, 1, 2])
            test_indices = np.array([3])
            mock_load_splits.return_value = (train_indices, test_indices)

            train_model(model_type='logreg', feature_set='structured')

            mock_load_splits.assert_called_once_with('structured', 'logreg')

            assert mock_joblib_dump.call_count == 2

    def test_train_model_split_not_found(self, mock_load_features):
        """Test that the function raises FileNotFoundError when split files don't exist."""
        with patch('src.models.train_model.load_data_splits', side_effect=FileNotFoundError("Split files not found")):
            with pytest.raises(FileNotFoundError, match="Split files not found"):
                train_model(model_type='logreg', feature_set='structured')
