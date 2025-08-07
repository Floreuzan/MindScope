import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import joblib
import matplotlib.pyplot as plt
import shap

from src.models.explain import explain_model

class TestExplainModel:
    """Test suite for model explanation functionality."""

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
        model = MagicMock()
        return model

    @pytest.fixture
    def mock_joblib_load(self, mock_model):
        """Fixture to mock joblib.load function."""
        with patch('src.models.explain.joblib.load') as mock:
            mock.return_value = mock_model
            yield mock

    @pytest.fixture
    def mock_load_features(self, mock_features):
        """Fixture to mock the load_features function."""
        with patch('src.models.explain.load_features') as mock:
            mock.return_value = mock_features
            yield mock

    @pytest.fixture
    def mock_makedirs(self):
        """Fixture to mock os.makedirs function."""
        with patch('src.models.explain.os.makedirs') as mock:
            yield mock

    @pytest.fixture
    def mock_shap_explainer(self):
        """Fixture to mock shap.Explainer."""
        with patch('src.models.explain.shap.Explainer') as mock:
            mock_explainer = MagicMock()
            mock_explainer.return_value = MagicMock()
            mock.return_value = mock_explainer
            yield mock

    @pytest.fixture
    def mock_shap_summary_plot(self):
        """Fixture to mock shap.summary_plot."""
        with patch('src.models.explain.shap.summary_plot') as mock:
            yield mock

    @pytest.fixture
    def mock_plt(self):
        """Fixture to mock matplotlib.pyplot functions."""
        with patch('src.models.explain.plt.figure') as mock_figure, \
             patch('src.models.explain.plt.savefig') as mock_savefig, \
             patch('src.models.explain.plt.close') as mock_close:
            yield {
                'figure': mock_figure,
                'savefig': mock_savefig,
                'close': mock_close
            }

    def test_explain_model_structured(self, mock_joblib_load, mock_load_features, mock_makedirs, 
                                     mock_shap_explainer, mock_shap_summary_plot, mock_plt):
        """Test explaining a model with structured features."""
        explain_model(model_type='logreg', feature_set='structured')
        
        mock_joblib_load.assert_called_once_with('models/logreg_structured.joblib')
        
        mock_load_features.assert_called_once_with('structured')
        
        mock_makedirs.assert_called_once_with('results', exist_ok=True)
        
        mock_shap_explainer.assert_called_once()
        
        mock_shap_summary_plot.assert_called_once()
        
        mock_plt['figure'].assert_called_once_with(figsize=(12, 8))
        
        mock_plt['savefig'].assert_called_once_with(
            'results/shap_summary_logreg_structured.png', 
            bbox_inches='tight', 
            dpi=300
        )
        
        mock_plt['close'].assert_called_once()

    def test_explain_model_non_structured(self, mock_load_features, mock_joblib_load):
        """Test that explain_model returns early for non-structured features."""
        explain_model(model_type='logreg', feature_set='tfidf')
        
        mock_joblib_load.assert_not_called()
        
        mock_load_features.assert_not_called()

    @pytest.mark.parametrize("model_type", ['logreg', 'rf', 'svm', 'nn'])
    def test_explain_model_different_models(self, model_type, mock_joblib_load, mock_load_features, 
                                           mock_makedirs, mock_shap_explainer, mock_shap_summary_plot, mock_plt):
        """Test explaining different model types."""
        explain_model(model_type=model_type, feature_set='structured')
        
        mock_joblib_load.assert_called_once_with(f'models/{model_type}_structured.joblib')
        
        mock_plt['savefig'].assert_called_once_with(
            f'results/shap_summary_{model_type}_structured.png', 
            bbox_inches='tight', 
            dpi=300
        )