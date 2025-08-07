import os
import pandas as pd
import numpy as np
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

import matplotlib
import matplotlib.pyplot as plt

# Import the module to test
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.visualize_features import visualize_label_distribution, visualize_feature_correlations, visualize_feature_importance, visualize_pca_projection

class TestVisualizationFunctions:
    """Test suite for visualization functions."""

    @pytest.fixture
    def mock_feature_data(self):
        """Fixture to create mock feature data for testing."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'feature3': [10, 20, 30, 40],
            'label': [0, 1, 0, 1]
        })
        return df

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Fixture to set up directories for testing."""
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        return figures_dir

    def test_visualize_label_distribution(self, mock_feature_data, setup_dirs, monkeypatch):
        """Test label distribution visualization."""
        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)

        monkeypatch.setattr(plt, "show", lambda: None)

        output_file = setup_dirs / "label_distribution.png"
        visualize_label_distribution(mock_feature_data, output_file=str(output_file))

        mock_savefig.assert_called_once_with(str(output_file), bbox_inches='tight', dpi=300)

    def test_visualize_feature_correlations(self, mock_feature_data, setup_dirs, monkeypatch):
        """Test feature correlations visualization."""
        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)

        monkeypatch.setattr(plt, "show", lambda: None)

        output_file = setup_dirs / "feature_correlations.png"
        visualize_feature_correlations(mock_feature_data, output_file=str(output_file))

        mock_savefig.assert_called_once_with(str(output_file), bbox_inches='tight', dpi=300)

    def test_visualize_feature_importance(self, mock_feature_data, setup_dirs, monkeypatch):
        """Test feature importance visualization."""
        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)

        monkeypatch.setattr(plt, "show", lambda: None)

        output_file = setup_dirs / "feature_importance.png"
        visualize_feature_importance(mock_feature_data, output_file=str(output_file))

        mock_savefig.assert_called_once_with(str(output_file), bbox_inches='tight', dpi=300)

    def test_visualize_pca_projection(self, mock_feature_data, setup_dirs, monkeypatch):
        """Test PCA projection visualization."""
        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)

        monkeypatch.setattr(plt, "show", lambda: None)

        output_file = setup_dirs / "pca_projection.png"
        visualize_pca_projection(mock_feature_data, output_file=str(output_file))

        mock_savefig.assert_called_once_with(str(output_file), bbox_inches='tight', dpi=300)

    def test_visualization_with_empty_data(self, setup_dirs, monkeypatch):
        """Test visualization functions with empty data."""
        empty_df = pd.DataFrame(columns=['feature1', 'feature2', 'label'])

        mock_savefig = MagicMock()
        monkeypatch.setattr(plt, "savefig", mock_savefig)

        monkeypatch.setattr(plt, "show", lambda: None)

        with pytest.raises(ValueError, match="Empty dataframe"):
            visualize_label_distribution(empty_df, output_file=str(setup_dirs / "empty_label_dist.png"))

        with pytest.raises(ValueError, match="Empty dataframe"):
            visualize_feature_correlations(empty_df, output_file=str(setup_dirs / "empty_correlations.png"))

        with pytest.raises(ValueError, match="Empty dataframe"):
            visualize_feature_importance(empty_df, output_file=str(setup_dirs / "empty_importance.png"))

        with pytest.raises(ValueError, match="Empty dataframe"):
            visualize_pca_projection(empty_df, output_file=str(setup_dirs / "empty_pca.png"))

    @pytest.mark.integration
    def test_visualization_integration(self, tmp_path, monkeypatch):
        """Integration test for visualization functions with realistic data."""
        os.makedirs(tmp_path / "features/nlp", exist_ok=True)
        os.makedirs(tmp_path / "features/structured", exist_ok=True)
        os.makedirs(tmp_path / "reports/figures", exist_ok=True)

        structured_df = pd.DataFrame({
            "char_length": [100, 110, 90, 95],
            "word_count": [20, 22, 18, 19],
            "vader_sentiment": [0.5, -0.3, 0.1, 0.0],
            "label": [1, 0, 1, 0]
        })

        structured_df.to_csv(tmp_path / "features/structured/structured_features.csv", index=False)

        monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.setattr(plt, "close", lambda: None)

        with monkeypatch.context() as m:
            m.chdir(tmp_path)

            output_file = tmp_path / "reports/figures/label_distribution.png"
            visualize_label_distribution(structured_df, output_file=str(output_file))

            output_file = tmp_path / "reports/figures/feature_correlations.png"
            visualize_feature_correlations(structured_df, output_file=str(output_file))

            output_file = tmp_path / "reports/figures/feature_importance.png"
            visualize_feature_importance(structured_df, output_file=str(output_file))

            output_file = tmp_path / "reports/figures/pca_projection.png"
            visualize_pca_projection(structured_df, output_file=str(output_file))
