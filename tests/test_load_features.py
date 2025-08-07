import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from src.utils.load_features import load_features

class TestLoadFeatures:
    """Test suite for feature loading functionality."""

    @pytest.fixture
    def mock_structured_df(self):
        """Fixture to create a mock structured features DataFrame."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.1, 0.2, 0.3],
            'label': [0, 1, 0]
        })

    @pytest.fixture
    def mock_tfidf_df(self):
        """Fixture to create a mock TF-IDF features DataFrame."""
        return pd.DataFrame({
            'word1': [0.1, 0.2, 0.3],
            'word2': [0.4, 0.5, 0.6],
            'label': [0, 1, 0]
        })

    @pytest.fixture
    def mock_glove_df(self):
        """Fixture to create a mock GloVe features DataFrame."""
        return pd.DataFrame({
            'glove_1': [0.1, 0.2, 0.3],
            'glove_2': [0.4, 0.5, 0.6],
            'label': [0, 1, 0]
        })

    @pytest.fixture
    def mock_bert_df(self):
        """Fixture to create a mock BERT features DataFrame."""
        return pd.DataFrame({
            'bert_1': [0.1, 0.2, 0.3],
            'bert_2': [0.4, 0.5, 0.6],
            'label': [0, 1, 0]
        })

    @pytest.fixture
    def mock_read_csv(self, mock_structured_df, mock_tfidf_df, mock_glove_df, mock_bert_df):
        """Fixture to mock pd.read_csv function."""
        def side_effect(filename):
            if 'structured' in filename:
                return mock_structured_df
            elif 'tfidf' in filename:
                return mock_tfidf_df
            elif 'glove' in filename:
                return mock_glove_df
            elif 'bert' in filename:
                return mock_bert_df
            else:
                raise ValueError(f"Unexpected filename: {filename}")

        with patch('src.utils.load_features.pd.read_csv') as mock:
            mock.side_effect = side_effect
            yield mock

    def test_load_features_structured(self, mock_read_csv, mock_structured_df):
        """Test loading structured features."""
        X, y = load_features(feature_set='structured')

        mock_read_csv.assert_called_once_with('features/structured/structured_features.csv')

        assert X.shape == (3, 2)  # 3 rows, 2 feature columns
        assert y.shape == (3,)    # 3 rows

        assert set(X.columns) == {'feature1', 'feature2'}

        np.testing.assert_array_equal(y, mock_structured_df['label'])

    def test_load_features_tfidf(self, mock_read_csv, mock_tfidf_df):
        """Test loading TF-IDF features."""
        X, y = load_features(feature_set='tfidf')

        mock_read_csv.assert_called_once_with('features/nlp/tfidf.csv')

        assert X.shape == (3, 2)  # 3 rows, 2 feature columns
        assert y.shape == (3,)    # 3 rows

        assert set(X.columns) == {'word1', 'word2'}

        np.testing.assert_array_equal(y, mock_tfidf_df['label'])

    def test_load_features_glove(self, mock_read_csv, mock_glove_df):
        """Test loading GloVe features."""
        X, y = load_features(feature_set='glove')

        mock_read_csv.assert_called_once_with('features/nlp/glove.csv')

        assert X.shape == (3, 2)  # 3 rows, 2 feature columns
        assert y.shape == (3,)    # 3 rows

        assert set(X.columns) == {'glove_1', 'glove_2'}

        np.testing.assert_array_equal(y, mock_glove_df['label'])

    def test_load_features_bert(self, mock_read_csv, mock_bert_df):
        """Test loading BERT features."""
        X, y = load_features(feature_set='bert')

        mock_read_csv.assert_called_once_with('features/nlp/bert.csv')

        assert X.shape == (3, 2)  # 3 rows, 2 feature columns
        assert y.shape == (3,)    # 3 rows

        assert set(X.columns) == {'bert_1', 'bert_2'}

        np.testing.assert_array_equal(y, mock_bert_df['label'])

    def test_load_features_all(self, mock_read_csv, mock_structured_df, mock_tfidf_df, mock_glove_df, mock_bert_df):
        """Test loading all features."""
        X, y = load_features(feature_set='all')

        assert mock_read_csv.call_count == 4
        mock_read_csv.assert_any_call('features/structured/structured_features.csv')
        mock_read_csv.assert_any_call('features/nlp/tfidf.csv')
        mock_read_csv.assert_any_call('features/nlp/glove.csv')
        mock_read_csv.assert_any_call('features/nlp/bert.csv')

        expected_cols = len(mock_structured_df.columns) + len(mock_tfidf_df.columns) + len(mock_glove_df.columns) + len(mock_bert_df.columns) - 4  # -4 for the label columns
        assert X.shape == (3, expected_cols)
        assert y.shape == (3,)

        np.testing.assert_array_equal(y, mock_structured_df['label'])

    def test_load_features_invalid_set(self, mock_read_csv):
        """Test that an invalid feature set raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid feature set"):
            load_features(feature_set='invalid_set')

    def test_load_features_different_row_counts(self, mock_structured_df, mock_tfidf_df, mock_glove_df, mock_bert_df):
        """Test that a warning is issued when feature files have different row counts."""
        modified_tfidf_df = mock_tfidf_df.copy()
        modified_tfidf_df = modified_tfidf_df.iloc[:-1]

        def side_effect(filename):
            if 'structured' in filename:
                return mock_structured_df
            elif 'tfidf' in filename:
                return modified_tfidf_df
            elif 'glove' in filename:
                return mock_glove_df
            elif 'bert' in filename:
                return mock_bert_df
            else:
                raise ValueError(f"Unexpected filename: {filename}")

        with pytest.warns() as recorded_warnings:
            with patch('src.utils.load_features.pd.read_csv', side_effect=side_effect):
                X, y = load_features(feature_set='all')

                assert X.shape[0] == 3
                assert y.shape == (3,)

        warning_messages = [str(w.message) for w in recorded_warnings]
        assert any("Feature files have different numbers of rows" in msg for msg in warning_messages)
        assert any("Label columns in feature files are not identical" in msg for msg in warning_messages)

    def test_load_features_different_labels(self, mock_structured_df, mock_tfidf_df, mock_glove_df, mock_bert_df):
        """Test that a warning is issued when feature files have different label columns."""
        modified_tfidf_df = mock_tfidf_df.copy()
        modified_tfidf_df['label'] = [1, 0, 1]

        def side_effect(filename):
            if 'structured' in filename:
                return mock_structured_df
            elif 'tfidf' in filename:
                return modified_tfidf_df
            elif 'glove' in filename:
                return mock_glove_df
            elif 'bert' in filename:
                return mock_bert_df
            else:
                raise ValueError(f"Unexpected filename: {filename}")

        with pytest.warns(UserWarning, match="Label columns in feature files are not identical") as recorded_warnings:
            with patch('src.utils.load_features.pd.read_csv', side_effect=side_effect):
                X, y = load_features(feature_set='all')

                assert X.shape[0] == 3
                assert y.shape == (3,)
