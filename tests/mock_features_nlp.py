import pandas as pd
import numpy as np

def mock_extract_bert_features(df, output_file, model_name='distilbert-base-uncased', train_indices=None, test_indices=None):
    """
    A simplified version of extract_bert_features for testing purposes.
    """
    bert_features = np.zeros((len(df), 768))
    columns = [f'bert_{i}' for i in range(768)]
    bert_df = pd.DataFrame(bert_features, columns=columns)
    
    # Add label column
    bert_df['label'] = df['label'].values
    
    # Save to CSV if output_file is provided
    if output_file is not None:
        bert_df.to_csv(output_file, index=False)
    
    return bert_df

def mock_extract_sentence_bert_features(df, output_file, model_name='all-MiniLM-L6-v2', train_indices=None, test_indices=None):
    """
    A simplified version of extract_sentence_bert_features for testing purposes.
    """
    sbert_features = np.zeros((len(df), 384))
    columns = [f'sbert_{i}' for i in range(384)]
    sbert_df = pd.DataFrame(sbert_features, columns=columns)
    
    sbert_df['label'] = df['label'].values
    
    if output_file is not None:
        sbert_df.to_csv(output_file, index=False)
    
    return sbert_df