import argparse
from src.models.train_model import train_model
from src.models.evaluate import evaluate_model
from src.models.explain import explain_model
from src.data.create_splits import create_data_splits

def main():
    parser = argparse.ArgumentParser(description="Run mental health risk model pipeline")
    parser.add_argument('--stage', type=str, choices=['split', 'train', 'evaluate', 'explain'], required=True)
    parser.add_argument('--model', type=str, default='logreg')
    parser.add_argument('--features', type=str, choices=['structured', 'tfidf', 'bert', 'glove', 'all'], default='all')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.stage == 'split':
        create_data_splits(feature_set=args.features, model_type=args.model, test_size=args.test_size, random_state=args.random_state)
    elif args.stage == 'train':
        train_model(model_type=args.model, feature_set=args.features)
    elif args.stage == 'evaluate':
        evaluate_model(model_type=args.model, feature_set=args.features)
    elif args.stage == 'explain':
        explain_model(model_type=args.model, feature_set=args.features)

if __name__ == "__main__":
    main()
