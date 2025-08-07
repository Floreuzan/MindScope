import numpy as np
import pandas as pd
import pytest


def test_clean_text_removes_noise():
    from src.preprocessing import clean_text
    noisy = "Check this out!!! 😢 @user http://test.com"
    cleaned = clean_text(noisy)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "😢" not in cleaned
    assert isinstance(cleaned, str)

@pytest.mark.parametrize("text", ["", "!!!", np.nan])
def test_clean_text_edge_cases(text):
    from src.preprocessing import clean_text
    result = clean_text(str(text))
    assert isinstance(result, str)

@pytest.mark.integration
def test_preprocess_dataset_columns(preprocessing_test_df, tmp_path):
    from src.preprocessing import preprocess_dataset
    in_file = tmp_path / "input.csv"
    out_file = tmp_path / "output.csv"
    preprocessing_test_df.to_csv(in_file, index=False)

    preprocess_dataset(in_file, out_file)
    df_out = pd.read_csv(out_file)
    assert "cleaned_text" in df_out.columns
    assert df_out.shape[0] == 2
