Pipeline Overview
Load the dataset and inspect relevant columns.
Preprocess text data (cleaning, tokenization, stopword removal, etc.).
Convert text into numerical format (TF-IDF, Word Embeddings, etc.).
Train a classification model (e.g., Logistic Regression, Random Forest, or an advanced deep-learning model).
Evaluate the model on a test set.
Make predictions on unseen reviews.


Class Imbalance Issue

Some wine varieties are underrepresented (e.g., Albariño).
Solution: SMOTE (Synthetic Oversampling) or Class Weights.
Feature Engineering Could Improve Performance

Adding N-Grams (bi-grams, tri-grams) might help better capture word context.
LLM-based embeddings (e.g., BERT, OpenAI) could improve representation.
Hyperparameter Tuning

Experiment with different TF-IDF settings (e.g., max_df, min_df).
Optimize Random Forest parameters (e.g., n_estimators, max_depth).


tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Keep feature size manageable
    ngram_range=(1,2),  # Include unigrams & bigrams
    stop_words="english"  # Remove common stopwords
)
