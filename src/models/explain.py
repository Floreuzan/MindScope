import joblib
import shap
from src.utils.load_features import load_features
import matplotlib.pyplot as plt
import os

def explain_model(model_type='logreg', feature_set='structured'):
    os.makedirs('results', exist_ok=True)
    if feature_set != 'structured':
        print("SHAP currently supports structured features only")
        return

    model = joblib.load(f'models/{model_type}_{feature_set}.joblib')
    X, y = load_features(feature_set)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)

    output_file = f'results/shap_summary_{model_type}_{feature_set}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"SHAP summary plot saved to {output_file}")
