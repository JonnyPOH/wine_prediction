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



concatinate the region maybe




API
> User/API Client sends a request (wine characteristics) via API Gateway
> API Gateway forwards the request to an AWS Lambda function
> Lambda
  - process
  - loads model
  - returns predictions
> Response is sent back to the user via API Gateway


aws lambda create-function --function-name predict-wine-country \
    --runtime python3.8 \
    --role arn:aws:iam::637423214227:role/Lambda_WinePrediction_Role \
    --handler lambda_function.lambda_handler \
    --timeout 30 \
    --memory-size 256 \
    --zip-file fileb://function.zip



Fine-Tuning Process

The model is initialized with pre-trained weights from EleutherAI/gpt-neo-1.3B.
The training script fine-tunes the model using:
Loss function (e.g., cross-entropy for text generation).
Backpropagation to adjust model weights.
Batch size of 8 (each step processes 8 samples).
3 epochs (model sees the full dataset 3 times)
