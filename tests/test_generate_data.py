def test_generate_dataset_output_shape():
    from src.generate_data import generate_dataset
    df = generate_dataset(num_samples=100)
    assert df.shape[0] == 100
    assert "text" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})

def test_label_proportion():
    from src.generate_data import generate_dataset
    df = generate_dataset(num_samples=1000)
    high_ratio = df['label'].mean()
    assert 0.40 < high_ratio < 0.50
