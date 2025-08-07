import sys
import os
import pytest
import pandas as pd
import numpy as np
import torch
import warnings
from unittest.mock import patch, MagicMock

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1.*")
warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)
warnings.filterwarnings("ignore", message="Label columns in feature files are not identical.*")
warnings.filterwarnings("ignore", message="Feature files have different numbers of rows.*")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.features_nlp import extract_tfidf_features, extract_glove_features
try:
    from src.features_nlp import extract_bert_features, extract_sentence_bert_features
except ImportError:
    extract_bert_features = MagicMock(return_value=pd.DataFrame())
    extract_sentence_bert_features = MagicMock(return_value=pd.DataFrame())

@pytest.fixture
def tfidf_test_df():
    """Create a test dataframe for TF-IDF feature extraction."""
    return pd.DataFrame({
        "cleaned_text": ["This is a test", "Another test document", "Third test sample"],
        "label": [0, 1, 0]
    })

@pytest.fixture
def glove_test_df():
    """Create a test dataframe for GloVe feature extraction."""
    return pd.DataFrame({
        "text": ["This is a test", "Another test document", "Third test sample"],
        "cleaned_text": ["This is a test", "Another test document", "Third test sample"],
        "label": [0, 1, 0]
    })

@pytest.fixture
def preprocessing_test_df():
    """Create a test dataframe for preprocessing."""
    return pd.DataFrame({
        "text": ["This is a test", "Another test document"],
        "cleaned_text": ["This is a test", "Another test document"],
        "label": [0, 1]
    })

@pytest.fixture
def structured_basic_df():
    """Create a test dataframe for structured feature extraction."""
    return pd.DataFrame({
        "text": ["This is a test", "Another test document", "Third test sample"],
        "cleaned_text": ["This is a test", "Another test document", "Third test sample"],
        "label": [0, 1, 0],
        "timestamp": ["2023-01-01 12:00:00", "2023-01-02 13:00:00", "2023-01-03 14:00:00"]
    })

@pytest.fixture
def create_edge_case_df():
    """Create a function that returns a dataframe with a single row containing the given text."""
    def _create_df(text):
        return pd.DataFrame({
            "text": [text],
            "cleaned_text": [text],
            "label": [0],
            "timestamp": ["2023-01-01 12:00:00"]
        })
    return _create_df

@pytest.fixture
def temp_output_file(tmp_path):
    """Create a temporary output file path."""
    def _temp_output_file(filename):
        return os.path.join(tmp_path, filename)
    return _temp_output_file


class TestTFIDFFeatures:
    def test_tfidf_extraction_basic(self, tfidf_test_df, temp_output_file):
        output_file = temp_output_file("tfidf_features.csv")
        result_df = extract_tfidf_features(df=tfidf_test_df, output_file=output_file)
        assert result_df.shape[0] == tfidf_test_df.shape[0], "Output should have same number of rows as input"
        assert any("label" in col for col in result_df.columns), "Label column should be preserved"
        assert result_df.shape[1] > 10, "Should extract multiple TF-IDF features"
        assert os.path.exists(output_file), "Output file should be created"

    @pytest.mark.parametrize("max_features,ngram_range", [
        (10, (1, 1)),
        (20, (1, 2)),
        (50, (1, 3)),
    ])
    def test_tfidf_parameters(self, tfidf_test_df, temp_output_file, max_features, ngram_range):
        output_file = temp_output_file(f"tfidf_features_{max_features}_{ngram_range[1]}.csv")
        result_df = extract_tfidf_features(
            df=tfidf_test_df,
            output_file=output_file,
            max_features=max_features,
            ngram_range=ngram_range
        )
        expected_cols = max_features + 1
        assert result_df.shape[1] <= expected_cols, f"Should have at most {expected_cols} columns"

    def test_tfidf_empty_input(self, temp_output_file):
        empty_df = pd.DataFrame({"cleaned_text": [], "label": []})
        output_file = temp_output_file("empty_tfidf.csv")
        with pytest.raises(ValueError, match="Empty input"):
            extract_tfidf_features(df=empty_df, output_file=output_file)


class TestGloVeFeatures:
    def test_glove_output_dims(self, glove_test_df):
        glove_test_df["text"] = glove_test_df["cleaned_text"]
        features = extract_glove_features(glove_test_df)
        assert features.shape[0] == glove_test_df.shape[0], "Should preserve number of rows"
        assert features.shape[1] >= 100, "Should have at least 100 dimensions (GloVe features + label)"
        assert "label" in features.columns, "Label column should be preserved"
        feature_cols = [col for col in features.columns if col != "label"]
        assert all(features[feature_cols].dtypes == np.float64), "Feature columns should be float64"

    def test_glove_handles_unknown_words(self, create_edge_case_df):
        df_with_unknown = create_edge_case_df("xyzabc123 is not a real word")
        df_with_unknown["text"] = df_with_unknown["cleaned_text"]
        features = extract_glove_features(df_with_unknown)
        assert features.shape[0] == 1, "Should have one row"
        assert features.shape[1] >= 100, "Should have expected number of dimensions"

    def test_glove_with_train_test_indices(self, glove_test_df, tmp_path):
        glove_test_df["text"] = glove_test_df["cleaned_text"]
        train_indices = pd.DataFrame({'index': [0, 1]})
        test_indices = pd.DataFrame({'index': [2]})
        train_file = tmp_path / "train_indices.csv"
        test_file = tmp_path / "test_indices.csv"
        train_indices.to_csv(train_file, index=False)
        test_indices.to_csv(test_file, index=False)
        features = extract_glove_features(
            glove_test_df,
            train_indices=train_file,
            test_indices=test_file
        )
        assert features.shape[0] == glove_test_df.shape[0], "Should preserve number of rows"
        assert features.shape[1] >= 100, "Should have at least 100 dimensions (GloVe features + label)"


class TestBERTFeatures:
    @pytest.mark.parametrize("model_name", ["distilbert-base-uncased"])
    def test_bert_output_dims(self, glove_test_df, temp_output_file, model_name):
        output_file = temp_output_file("bert_features.csv")
        glove_test_df["text"] = glove_test_df.get("text", glove_test_df["cleaned_text"])
        glove_test_df["cleaned_text"] = glove_test_df["text"]
        with patch('src.features_nlp.AutoModel.from_pretrained') as mock_model, \
             patch('src.features_nlp.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.return_value = {"input_ids": torch.ones((1, 10)), "attention_mask": torch.ones((1, 10))}
            mock_tokenizer.return_value = mock_tokenizer_instance
            mock_model_instance = MagicMock()
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.zeros((glove_test_df.shape[0], 10, 768))
            mock_model_instance.return_value = mock_output
            mock_model.return_value = mock_model_instance
            features = extract_bert_features(
                df=glove_test_df,
                output_file=output_file,
                model_name=model_name
            )
            assert features is not None, "Features should not be None"
            assert features.shape[0] == glove_test_df.shape[0], "Should preserve number of rows"
            assert "label" in features.columns, "Label column should be preserved"


class TestSentenceBERTFeatures:
    @pytest.mark.parametrize("model_name", ["all-MiniLM-L6-v2"])
    def test_sentence_bert_output_dims(self, glove_test_df, temp_output_file, model_name):
        output_file = temp_output_file("sentence_bert_features.csv")
        glove_test_df["text"] = glove_test_df.get("text", glove_test_df["cleaned_text"])
        glove_test_df["cleaned_text"] = glove_test_df["text"]
        with patch('src.features_nlp.SentenceTransformer') as mock_model:
            mock_model_instance = MagicMock()
            mock_model_instance.encode.return_value = np.zeros((glove_test_df.shape[0], 384))
            mock_model.return_value = mock_model_instance
            features = extract_sentence_bert_features(
                df=glove_test_df,
                output_file=output_file,
                model_name=model_name
            )
            assert features is not None, "Features should not be None"
            assert features.shape[0] == glove_test_df.shape[0], "Should preserve number of rows"
            assert "label" in features.columns, "Label column should be preserved"


@pytest.mark.integration
def test_full_nlp_pipeline_integration(preprocessing_test_df, temp_output_file):
    preprocessing_test_df["text"] = preprocessing_test_df.get("text", preprocessing_test_df["cleaned_text"])
    preprocessing_test_df["cleaned_text"] = preprocessing_test_df["text"]
    tfidf_output = temp_output_file("integration_tfidf.csv")
    bert_output = temp_output_file("integration_bert.csv")
    sentence_bert_output = temp_output_file("integration_sentence_bert.csv")
    tfidf_features = extract_tfidf_features(preprocessing_test_df, output_file=tfidf_output)
    glove_features = extract_glove_features(preprocessing_test_df)
    with patch('src.features_nlp.AutoModel.from_pretrained') as mock_bert_model, \
         patch('src.features_nlp.AutoTokenizer.from_pretrained') as mock_bert_tokenizer, \
         patch('src.features_nlp.SentenceTransformer') as mock_sentence_model:
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {"input_ids": torch.ones((1, 10)), "attention_mask": torch.ones((1, 10))}
        mock_bert_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.zeros((preprocessing_test_df.shape[0], 10, 768))
        mock_model_instance.return_value = mock_output
        mock_bert_model.return_value = mock_model_instance
        mock_sentence_model_instance = MagicMock()
        mock_sentence_model_instance.encode.return_value = np.zeros((preprocessing_test_df.shape[0], 384))
        mock_sentence_model.return_value = mock_sentence_model_instance
        bert_features = extract_bert_features(
            df=preprocessing_test_df,
            output_file=bert_output,
            model_name="distilbert-base-uncased"
        )
        sentence_bert_features = extract_sentence_bert_features(
            df=preprocessing_test_df,
            output_file=sentence_bert_output,
            model_name="all-MiniLM-L6-v2"
        )
        assert tfidf_features.shape[0] == preprocessing_test_df.shape[0]
        assert glove_features.shape[0] == preprocessing_test_df.shape[0]
        assert bert_features.shape[0] == preprocessing_test_df.shape[0]
        assert sentence_bert_features.shape[0] == preprocessing_test_df.shape[0]
        assert "label" in tfidf_features.columns
        assert "label" in glove_features.columns
        assert "label" in bert_features.columns
        assert "label" in sentence_bert_features.columns
        assert os.path.exists(tfidf_output)
        assert os.path.exists(bert_output)
        assert os.path.exists(sentence_bert_output)
