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




TODO:
> Sort our the lambda function and API code
> Deploying API Gateway to expose it as a public API.
> Integrating DynamoDB to store past predictions, test with curl
  (curl -X POST "https://xyz123.execute-api.us-east-1.amazonaws.com/prod/" \
     -H "Content-Type: application/json" \
     -d '{"wine_characteristics": "13.5 alcohol content, 2.8 acidity, and rich fruity aroma"}')

sat/sun, cleaning up and further fine tuning and improving models
sat/sun, drawing up an AWS architecture diagram/s
