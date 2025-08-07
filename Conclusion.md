# Exploratory Data Analysis Report: Mental Health Risk Classification

This report presents a comprehensive analysis of the structured and NLP-based features extracted from a synthetic dataset of 800 mental health-related text entries. Each entry has been labeled as either low-risk (label = 0) or high-risk (label = 1). The analysis integrates statistical, temporal, semantic, and lexical features to understand the underlying distributions and their relationship to mental health risk.

---

## Label Distribution

The label distribution reveals a class imbalance:

* **Low Risk (label = 0):** \~70% of samples
* **High Risk (label = 1):** \~30% of samples

This imbalance is moderate but important. It will influence model training and should be addressed with stratified splitting and potentially class-weighted loss functions or resampling strategies like SMOTE.

---

## Structured Feature Analysis

### 1. Post Length Metrics

* **Character Length & Word Count:**

  * High-risk posts are shorter, often concise and emotionally dense.
  * Low-risk posts are more variable, often longer and elaborative.

* **Average Sentence Length:**

  * High-risk posts exhibit shorter sentence structures.
  * Suggests urgency, fragmentation, or emotional strain in high-risk language.

### 2. Sentiment Features

* **TextBlob & VADER Sentiment:**

  * High-risk entries skew toward negative sentiment scores.
  * Low-risk entries are more neutral or slightly positive.
  * Confirms the emotional tone as a powerful indicator.

### 3. Temporal Features

* **Hour of Day:**

  * Subtle increase in high-risk posts during night/early morning hours.
  * Aligns with circadian rhythm disruptions seen in depressive or anxious individuals.

### 4. Stylometric & Linguistic Features

* Features like pronoun usage, negation words, and emotional tone ratios highlight writing style:

  * **High-risk posts:** Higher negative word usage, more first-person singular pronouns.
  * **Low-risk posts:** Richer in sentence complexity and less frequent use of self-reference.

---

## Feature Correlation Insights

A full correlation heatmap reveals:

* **High collinearity** among `char_length`, `word_count`, and `avg_sentence_length`, suggesting these could be reduced via PCA or dropped in favor of one.
* **VADER polarity features** (pos, neg, neu) are mutually inverse, as expected.
* **Emotional tone and sentiment variables** do not strongly correlate with stylometric or temporal features, indicating they provide **complementary signals**.

These insights support feature diversity and discourage excessive pruning.

---

## NLP Feature Projections

### 1. TF-IDF (PCA Visualization)

* Clear class separation along primary components.
* Indicates lexical term frequency is informative of mental health risk.

### 2. GloVe (PCA Visualization)

* Semantic vector averaging still reveals separability.
* Dense representations of meaning provide deeper patterns than frequency alone.

### 3. BERT (t-SNE Projection)

* Excellent separation of low-risk and high-risk posts in 2D space.
* Context-aware, transformer-derived embeddings encode nuance, tone, and structure very effectively.

These results validate the use of transformer models like BERT for text classification in emotionally charged domains.

---

## Interpretation & Implications

| Insight                                        | Interpretation                                 | Modeling Impact                              |
| ---------------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| High-risk posts are shorter and more negative  | Users in distress write less and more directly | Use length and sentiment as primary features |
| Semantic embeddings show label clustering      | Word choice and context matter                 | Use transformer models (e.g. BERT)           |
| Structured features capture orthogonal signals | Time, syntax, and sentiment add value          | Combine structured + NLP for hybrid models   |
| Class imbalance is present                     | Low-risk dominates in count                    | Use stratified sampling, class weights       |

---

## 
Next Steps & Recommendations

1. **Feature Engineering**

   * Normalize structured features
   * Drop redundant length features or apply PCA
   * Fuse structured + NLP into unified input vector

2. **Modeling Strategy**

   * Train baseline Logistic Regression / SVM on structured features
   * Compare with Random Forest / Feedforward NN on BERT embeddings
   * Evaluate ROC, AUC, F1, confusion matrix
   * Use bootstrapped confidence intervals for metric robustness

3. **Explainability**

   * Use SHAP on structured models to visualize key contributors
   * For BERT, attention weights or integrated gradients can offer limited interpretability

4. **Reproducibility**

   * All outputs tracked via DVC
   * Random seeds, versioned datasets, and modular scripts included

5. **Ethical Considerations**

   * Avoid overfitting to emotionally sensitive cues
   * Models must not trigger clinical actions without human validation
   * Outputs should be explainable and non-alarming to users

---

## Conclusion

This exploratory analysis demonstrates that structured metadata and modern NLP embeddings both offer powerful signals for classifying mental health risk. BERT-based sentence embeddings and sentiment-aware structured features complement each other effectively. Visual inspection confirms clear separability between classes, laying the foundation for robust, interpretable, and ethically informed classification models.

This EDA establishes the data fidelity, insight quality, and feature richness necessary to proceed confidently to the modeling phase.

## Pipeline Improvements: Fixing the Perfect Accuracy Issue

### Problem Identified
The machine learning pipeline was consistently achieving an accuracy of 1 (perfect accuracy), which is unrealistic for most real-world classification tasks. This indicated a potential issue with data leakage or overfitting.

### Root Causes
After examining the code, the following issues were identified:

1. **Data Leakage in Evaluation**: Both the training and evaluation code were using the same random seed (42) for the `train_test_split` function. This meant that the model was being evaluated on the exact same data it was trained on, leading to artificially high accuracy.

2. **Inconsistent Train/Test Splits**: The training code was splitting the data and training the model on the training set, but the evaluation code was creating a new split of the entire dataset. This meant that some data points used for training were also being used for evaluation.

3. **Feature Extraction Data Leakage**: The feature extraction process, particularly for TF-IDF features, was fitting models on the entire dataset, including both training and test data. This meant that information from the test set was leaking into the training process.

### Changes Made

1. **Consistent Train/Test Splits**: 
   - Modified the training code to save the train/test split indices to CSV files
   - Updated the evaluation code to use these saved indices instead of creating a new split

2. **Proper Data Handling**:
   - Fixed a bug in the training code where `train_test_split` was called twice with the same random seed
   - Implemented a more robust way to split the data using indices

3. **Strict Consistency in Evaluation**:
   - Modified the evaluation code to raise an error if test indices are missing, instead of silently creating a new split
   - Added a clear message prompting re-training if the test split isn't found

4. **Feature Extraction Improvements**:
   - Modified the TF-IDF feature extraction to fit the vectorizer only on the training data
   - Updated the GloVe, BERT, and SentenceTransformers feature extraction to use the same train/test split
   - Added a command-line flag to use existing train/test split indices in the feature extraction process
   - Updated the DVC pipeline to use this flag

### Expected Outcome
With these changes, the model should now be evaluated on data it hasn't seen during training, and the feature extraction process should not leak information from the test set into the training process. This should result in a more realistic accuracy that reflects the model's true generalization performance.

### Why This Fixes the Issue
By ensuring that the model is evaluated on a completely separate set of data from what it was trained on, and by preventing information from the test set from leaking into the feature extraction process, we eliminate the data leakage that was causing the perfect accuracy. The model now has to generalize to unseen data, which is a more realistic test of its performance.
