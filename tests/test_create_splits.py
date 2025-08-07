import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.create_splits import create_data_splits, load_data_splits

class TestCreateSplits:
    """Test suite for data split creation and loading functionality."""

    @pytest.fixture
    def mock_features(self):
        """Fixture to create mock features for testing."""
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0]
        ])
        y = np.array([0, 1, 0, 1, 0])
        return X, y

    @pytest.fixture
    def mock_load_features(self, mock_features):
        """Fixture to mock the load_features function."""
        with patch('src.data.create_splits.load_features') as mock:
            mock.return_value = mock_features
            yield mock

    def test_create_data_splits(self, mock_load_features, tmp_path):
        """Test creating train/test splits."""
        with patch('src.data.create_splits.os.makedirs') as mock_makedirs:
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                train_indices, test_indices = create_data_splits(
                    feature_set='structured',
                    model_type='logreg',
                    test_size=0.2,
                    random_state=42
                )

                mock_load_features.assert_called_once_with('structured')

                mock_makedirs.assert_called_once_with('data/splits', exist_ok=True)

                assert mock_to_csv.call_count == 2

                assert len(train_indices) == 4  # 80% of 5
                assert len(test_indices) == 1   # 20% of 5
                assert set(train_indices).isdisjoint(set(test_indices))  # No overlap
                assert set(train_indices).union(set(test_indices)) == set(range(5))  # All indices used

    def test_load_data_splits(self, tmp_path):
        """Test loading train/test splits."""
        splits_dir = tmp_path / "data" / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        train_indices = pd.DataFrame({'index': [0, 1, 2, 3]})
        test_indices = pd.DataFrame({'index': [4]})
        
        train_file = splits_dir / "logreg_structured_train_indices.csv"
        test_file = splits_dir / "logreg_structured_test_indices.csv"
        
        train_indices.to_csv(train_file, index=False)
        test_indices.to_csv(test_file, index=False)
        
        with patch('src.data.create_splits.pd.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [train_indices, test_indices]
            
            train_idx, test_idx = load_data_splits(
                feature_set='structured',
                model_type='logreg'
            )
            
            assert mock_read_csv.call_count == 2
            
            np.testing.assert_array_equal(train_idx, train_indices['index'].values)
            np.testing.assert_array_equal(test_idx, test_indices['index'].values)
    
    def test_load_data_splits_file_not_found(self):
        """Test that load_data_splits raises FileNotFoundError when files don't exist."""
        with patch('src.data.create_splits.pd.read_csv', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="Split files not found"):
                load_data_splits(feature_set='structured', model_type='logreg')