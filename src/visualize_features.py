# src/visualize_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid", context="notebook", font_scale=1.2)

os.makedirs("reports/figures", exist_ok=True)

def map_labels_to_names(labels):
    """
    Map numeric labels (0, 1) to descriptive names ("low risk", "high risk").

    Args:
        labels: Series or array of numeric labels

    Returns:
        Series with descriptive label names
    """
    label_map = {0: "low risk", 1: "high risk"}
    return pd.Series([label_map.get(label, label) for label in labels])

def plot_label_distribution(df):
    plt.figure(figsize=(8, 6))
    df_plot = df.copy()
    df_plot["label"] = map_labels_to_names(df["label"])
    sns.countplot(x="label", hue="label", data=df_plot, palette="pastel", legend=False)
    plt.title("Label Distribution", fontsize=16)
    plt.xlabel("Mental Health Risk Level", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/figures/label_distribution.png")
    plt.close()

def plot_structured_distributions(df):
    features = ["char_length", "textblob_sentiment", "vader_sentiment", "hour_of_day", "avg_sentence_length"]
    df_plot = df.copy()
    df_plot["label"] = map_labels_to_names(df["label"])
    for feat in features:
        plt.figure(figsize=(8, 6))
        # Plot each label group manually for clear label mapping
        for label_name, color in zip(["low risk", "high risk"], ["#66c2a5", "#fc8d62"]):
            subset = df_plot[df_plot["label"] == label_name]
            sns.histplot(data=subset, x=feat, kde=True, element="step", label=label_name, color=color, stat="density",
                         linewidth=1.5)

        plt.title(f"Distribution of {feat}", fontsize=16)
        plt.xlabel(feat, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(title="Risk Level")
        plt.tight_layout()
        plt.savefig(f"reports/figures/{feat}_distribution.png")
        plt.close()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 12))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75}
    )
    plt.title("Feature Correlation Heatmap", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()


def plot_pca_embedding(X, y, label, filename):
    plt.figure(figsize=(8, 6))
    reducer = TruncatedSVD(n_components=2)
    X_reduced = reducer.fit_transform(X)
    # Map numeric labels to descriptive names
    y_names = map_labels_to_names(y)
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y_names, palette="Set2", alpha=0.7)
    plt.title(f"PCA of {label} Features", fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Risk Level")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}.png")
    plt.close()

def plot_tsne_embedding(X, y, label, filename):
    plt.figure(figsize=(8, 6))
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    # Map numeric labels to descriptive names
    y_names = map_labels_to_names(y)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_names, palette="Set1", alpha=0.6)
    plt.title(f"t-SNE of {label} Features", fontsize=16)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Risk Level")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{filename}.png")
    plt.close()

# Wrapper functions for tests
def visualize_label_distribution(df, output_file=None):
    """
    Visualize the distribution of labels in the dataset.

    Args:
        df (pandas.DataFrame): DataFrame containing the data with a 'label' column
        output_file (str, optional): Path to save the output figure. If None, uses default path.
    """
    if df.empty:
        raise ValueError("Empty dataframe provided")

    plt.figure(figsize=(8, 6))
    # Create a copy of the dataframe with descriptive label names
    df_plot = df.copy()
    df_plot["label"] = map_labels_to_names(df["label"])
    sns.countplot(x="label", hue="label", data=df_plot, palette="pastel", legend=False)
    plt.title("Label Distribution", fontsize=16)
    plt.xlabel("Mental Health Risk Level", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.savefig("reports/figures/label_distribution.png", bbox_inches='tight', dpi=300)
    plt.close()

def visualize_feature_correlations(df, output_file=None):
    """
    Visualize correlations between features.

    Args:
        df (pandas.DataFrame): DataFrame containing the features
        output_file (str, optional): Path to save the output figure. If None, uses default path.
    """
    if df.empty:
        raise ValueError("Empty dataframe provided")

    plt.figure(figsize=(14, 12))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75}
    )
    plt.title("Feature Correlation Heatmap", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.savefig("reports/figures/correlation_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

def visualize_feature_importance(df, output_file=None):
    """
    Visualize feature importance using a Random Forest classifier.

    Args:
        df (pandas.DataFrame): DataFrame containing the features and 'label' column
        output_file (str, optional): Path to save the output figure. If None, uses default path.
    """
    if df.empty:
        raise ValueError("Empty dataframe provided")

    df_display = df.copy()
    df_display["label"] = map_labels_to_names(df["label"])

    X = df.drop('label', axis=1)
    y = df['label']

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_names = X.columns

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances for Risk Level Prediction", fontsize=16)
    plt.bar(range(min(20, len(indices))), importances[indices][:20], align='center')
    plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices][:20], rotation=90)
    plt.ylabel("Importance Score", fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.savefig("reports/figures/feature_importance.png", bbox_inches='tight', dpi=300)
    plt.close()

def visualize_pca_projection(df, output_file=None):
    """
    Visualize PCA projection of the features.

    Args:
        df (pandas.DataFrame): DataFrame containing the features and 'label' column
        output_file (str, optional): Path to save the output figure. If None, uses default path.
    """
    if df.empty:
        raise ValueError("Empty dataframe provided")

    X = df.drop('label', axis=1)
    y = df['label']

    y_names = map_labels_to_names(y)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for label_name in ["low risk", "high risk"]:
        idx = y_names == label_name
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label_name, alpha=0.7)

    plt.title("PCA Projection", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Risk Level")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.savefig("reports/figures/pca_projection.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("📥 Loading data...")
    structured = pd.read_csv("features/structured/structured_features.csv")
    tfidf = pd.read_csv("features/nlp/tfidf.csv")
    glove = pd.read_csv("features/nlp/glove.csv")
    bert = pd.read_csv("features/nlp/bert.csv")

    labels = structured["label"]

    print("🏷️ Converting numeric labels to descriptive names...")

    print("📊 Plotting label distribution...")
    plot_label_distribution(structured)

    print("📊 Plotting structured feature distributions...")
    plot_structured_distributions(structured)

    print("📊 Plotting correlation heatmap...")
    plot_correlation_heatmap(structured.drop(columns=["label"], errors="ignore"))

    print("🧬 Visualizing PCA and t-SNE embeddings...")
    plot_pca_embedding(tfidf, labels, "TF-IDF", "tfidf_pca")
    plot_pca_embedding(glove, labels, "GloVe", "glove_pca")
    plot_tsne_embedding(bert, labels, "BERT", "bert_tsne")

    print("✅ All figures saved to: reports/figures/")
