import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.features_structured import extract_structured_features, extract_post_length_features, extract_sentiment_features, extract_temporal_features, extract_stylometric_features


class TestPostLengthFeatures:
    """Test suite for post length feature extraction."""

    def test_post_length_basic(self, structured_basic_df):
        """Test basic post length feature extraction."""
        result = extract_post_length_features(structured_basic_df)

        assert "char_length" in result.columns
        assert "word_count" in result.columns

        assert result["char_length"].iloc[0] > 0
        assert result["word_count"].iloc[0] > 0

        assert result.shape[0] == structured_basic_df.shape[0]
        assert result.shape[1] == 2

    @pytest.mark.parametrize("text,expected_char_length,expected_word_count", [
        ("Hello world", 11, 2),
        ("", 0, 0),
        ("   ", 3, 0),
        ("One", 3, 1)
    ])
    def test_post_length_specific_cases(self, text, expected_char_length, expected_word_count, create_edge_case_df):
        """Test post length features with specific input cases."""
        df = create_edge_case_df(text)
        result = extract_post_length_features(df)

        assert result["char_length"].iloc[0] == expected_char_length
        assert result["word_count"].iloc[0] == expected_word_count


class TestSentimentFeatures:
    """Test suite for sentiment feature extraction."""

    def test_sentiment_basic(self, structured_basic_df):
        """Test basic sentiment feature extraction."""
        result = extract_sentiment_features(structured_basic_df)

        assert "textblob_sentiment" in result.columns
        assert "vader_sentiment" in result.columns
        assert "vader_pos" in result.columns
        assert "vader_neg" in result.columns
        assert "vader_neu" in result.columns

        assert result.shape[0] == structured_basic_df.shape[0]
        assert result.shape[1] == 5

    def test_sentiment_positive_text(self, create_edge_case_df):
        """Test sentiment features with positive text."""
        df = create_edge_case_df("I am very happy and excited about this project.")
        result = extract_sentiment_features(df)

        assert result["textblob_sentiment"].iloc[0] > 0
        assert result["vader_sentiment"].iloc[0] > 0
        assert result["vader_pos"].iloc[0] > result["vader_neg"].iloc[0]

    def test_sentiment_negative_text(self, create_edge_case_df):
        """Test sentiment features with negative text."""
        df = create_edge_case_df("I am very sad and disappointed about this failure.")
        result = extract_sentiment_features(df)

        assert result["textblob_sentiment"].iloc[0] < 0
        assert result["vader_sentiment"].iloc[0] < 0
        assert result["vader_neg"].iloc[0] > result["vader_pos"].iloc[0]

    def test_sentiment_neutral_text(self, create_edge_case_df):
        """Test sentiment features with neutral text."""
        df = create_edge_case_df("This is a simple statement of fact.")
        result = extract_sentiment_features(df)

        assert abs(result["textblob_sentiment"].iloc[0]) < 0.3
        assert abs(result["vader_sentiment"].iloc[0]) < 0.3
        assert result["vader_neu"].iloc[0] > 0.5


class TestTemporalFeatures:
    """Test suite for temporal feature extraction."""

    def test_temporal_basic(self, structured_basic_df):
        """Test basic temporal feature extraction."""
        result = extract_temporal_features(structured_basic_df)

        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
        assert "time_of_day" in result.columns

        assert result.shape[0] == structured_basic_df.shape[0]
        assert result.shape[1] == 5

    @pytest.mark.parametrize("timestamp,expected_hour,expected_day,expected_is_weekend", [
        ("2023-01-01 12:00", 12, 6, 1),  # Sunday
        ("2023-01-02 00:00", 0, 0, 0),   # Monday
        ("2023-01-06 18:30", 18, 4, 0),  # Friday
        ("2023-01-07 23:59", 23, 5, 1)   # Saturday
    ])
    def test_temporal_specific_timestamps(self, timestamp, expected_hour, expected_day, expected_is_weekend):
        """Test temporal features with specific timestamps."""
        df = pd.DataFrame({
            "cleaned_text": ["Test text"],
            "label": [0],
            "timestamp": [timestamp]
        })
        result = extract_temporal_features(df)

        assert result["hour_of_day"].iloc[0] == expected_hour
        assert result["day_of_week"].iloc[0] == expected_day
        assert result["is_weekend"].iloc[0] == expected_is_weekend


class TestStylometricFeatures:
    """Test suite for stylometric feature extraction."""

    def test_stylometric_basic(self, structured_basic_df):
        """Test basic stylometric feature extraction."""
        result = extract_stylometric_features(structured_basic_df)

        assert "avg_sentence_length" in result.columns
        assert "pronoun_percentage" in result.columns
        assert "first_person_singular_percentage" in result.columns
        assert "negation_word_count" in result.columns
        assert "positive_word_count" in result.columns
        assert "negative_word_count" in result.columns
        assert "positive_word_percentage" in result.columns
        assert "negative_word_percentage" in result.columns
        assert "emotional_tone_ratio" in result.columns

        assert result.shape[0] == structured_basic_df.shape[0]
        assert result.shape[1] == 9  # 9 stylometric features

    def test_stylometric_first_person(self, create_edge_case_df):
        """Test stylometric features with first-person text."""
        df = create_edge_case_df("I think I am feeling good about myself and my progress.")
        result = extract_stylometric_features(df)

        assert result["pronoun_percentage"].iloc[0] > 0
        assert result["first_person_singular_percentage"].iloc[0] > 0

    def test_stylometric_negation(self, create_edge_case_df):
        """Test stylometric features with negation words."""
        df = create_edge_case_df("I don't think this is not good. I can't believe it.")
        result = extract_stylometric_features(df)

        assert result["negation_word_count"].iloc[0] > 0

    def test_stylometric_emotional_tone(self, create_edge_case_df):
        """Test stylometric features with emotional tone."""
        df_pos = create_edge_case_df("happy excited wonderful amazing great")
        result_pos = extract_stylometric_features(df_pos)

        df_neg = create_edge_case_df("sad depressed anxious worried terrible")
        result_neg = extract_stylometric_features(df_neg)

        assert result_pos["positive_word_count"].iloc[0] > result_pos["negative_word_count"].iloc[0]

        assert result_neg["negative_word_count"].iloc[0] > result_neg["positive_word_count"].iloc[0]

        assert result_pos["emotional_tone_ratio"].iloc[0] > result_neg["emotional_tone_ratio"].iloc[0]

    def test_stylometric_empty_text(self, create_edge_case_df):
        """Test stylometric features with empty text."""
        df = create_edge_case_df("")
        result = extract_stylometric_features(df)

        assert result["avg_sentence_length"].iloc[0] == 0
        assert result["pronoun_percentage"].iloc[0] == 0
        assert result["first_person_singular_percentage"].iloc[0] == 0
        assert result["negation_word_count"].iloc[0] == 0
        assert result["positive_word_count"].iloc[0] == 0
        assert result["negative_word_count"].iloc[0] == 0


class TestStructuredFeatures:
    """Test suite for the full structured feature extraction pipeline."""

    def test_extract_structured_basic(self, structured_basic_df, tmp_path):
        """Test basic functionality of extract_structured_features."""
        output_file = tmp_path / "structured_features.csv"
        result = extract_structured_features(df=structured_basic_df, output_file=str(output_file))

        assert "char_length" in result.columns
        assert "vader_sentiment" in result.columns
        assert "hour_of_day" in result.columns
        assert "avg_sentence_length" in result.columns
        assert "label" in result.columns

        assert output_file.exists()

        assert result.shape[0] == structured_basic_df.shape[0]
        assert result.shape[1] > 10

    @pytest.mark.parametrize("text", ["", "     ", "unknown emotionless entry"])
    def test_structured_features_text_edge_cases(self, text, create_edge_case_df, tmp_path):
        """Test extract_structured_features with edge case texts."""
        df = create_edge_case_df(text)
        output_file = tmp_path / "edge_case_features.csv"
        out = extract_structured_features(df=df, output_file=str(output_file))

        assert out.shape[0] == 1
        assert out.shape[1] > 10

        assert output_file.exists()

    def test_extract_structured_no_output_file(self, structured_basic_df):
        """Test extract_structured_features without specifying an output file."""
        result = extract_structured_features(df=structured_basic_df, output_file=None)

        assert isinstance(result, pd.DataFrame)
        assert "char_length" in result.columns
        assert "label" in result.columns

    def test_extract_structured_input_validation(self):
        """Test that extract_structured_features validates input correctly."""
        with pytest.raises(ValueError, match="Either df or input_file must be provided"):
            extract_structured_features()

    @pytest.mark.integration
    def test_structured_features_integration(self, structured_basic_df, tmp_path):
        """Integration test for the full structured feature extraction pipeline."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        structured_basic_df.to_csv(input_file, index=False)

        result = extract_structured_features(input_file=str(input_file), output_file=str(output_file))

        assert output_file.exists()
        df_loaded = pd.read_csv(output_file)
        assert df_loaded.shape[0] == structured_basic_df.shape[0]
        assert "char_length" in df_loaded.columns
        assert "vader_sentiment" in df_loaded.columns
        assert "label" in df_loaded.columns
