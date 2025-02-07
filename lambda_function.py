import json
import boto3
import pickle
import os
import numpy as np
from io import BytesIO
from botocore.exceptions import NoCredentialsError

# Initialize AWS services
s3 = boto3.client("s3")
runtime = boto3.client("sagemaker-runtime")

# Constants
BUCKET_NAME = "wine-country-models"
MODEL_FILE_KEY = "wine_model.pkl"
# SAGEMAKER_ENDPOINT = "your-llm-endpoint-name"

# Load model from S3
def load_model():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_FILE_KEY)
        model_data = response["Body"].read()
        model = pickle.load(BytesIO(model_data))
        return model
    except NoCredentialsError:
        raise Exception("AWS credentials not found!")

# # Function to call SageMaker LLM
# def call_sagemaker_llm(input_text):
#     payload = json.dumps({"inputs": input_text})
#     response = runtime.invoke_endpoint(
#         EndpointName=SAGEMAKER_ENDPOINT,
#         ContentType="application/json",
#         Body=payload,
#     )
#     result = json.loads(response["Body"].read().decode("utf-8"))
#     return result.get("generated_text", "LLM response unavailable")

# Lambda entry point
def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        features = np.array(body["wine_characteristics"]).reshape(1, -1)
        model_choice = body.get("model", "ml")  # Choose 'ml' or 'llm'

        if model_choice == "ml":
            model = load_model()
            prediction = model.predict(features)
            response = {"predicted_country": prediction[0]}
        # else:
        #     input_text = body.get("text_query", "Predict wine country")
        #     response = {"llm_prediction": call_sagemaker_llm(input_text)}

        return {"statusCode": 200, "body": json.dumps(response)}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
