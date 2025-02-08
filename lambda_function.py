import json
import openai
import os
import boto3

# AWS Secrets Manager Client
secrets_client = boto3.client("secretsmanager")

# Correct Secret Name (Make sure this matches your secret in AWS Secrets Manager)
SECRET_NAME = "open_ai_secret_jonny_2"  # Ensure this matches the name in AWS Secrets Manager

def get_openai_api_key():
    """Retrieve OpenAI API key from AWS Secrets Manager."""
    try:
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        secret_string = json.loads(response["SecretString"])
        return secret_string["open_ai_secret_jonny_2"]  # Extract OpenAI key correctly
    except secrets_client.exceptions.ResourceNotFoundException:
        raise Exception(f"Secret '{SECRET_NAME}' not found in AWS Secrets Manager.")
    except Exception as e:
        raise Exception(f"Error retrieving secret: {str(e)}")

# Set OpenAI API Key securely
openai.api_key = get_openai_api_key()

# Initialize OpenAI client (new API)
client = openai.OpenAI(api_key=openai.api_key)

def lambda_handler(event, context):
    """AWS Lambda handler for wine country prediction."""
    try:
        # Log the incoming event for debugging
        print("Received event:", json.dumps(event))

        # Check if the body exists
        if "body" not in event:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'body' in request"})
            }

        # Parse request body correctly
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]

        # Extract wine characteristics
        wine_characteristics = body.get("wine_characteristics")
        if not wine_characteristics:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'wine_characteristics' in request"})
            }

        # Create the prompt for OpenAI model
        prompt = f"Predict the country of origin based on the following characteristics: {wine_characteristics}"

        # OpenAI API call using new syntax
        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::AyGh2lZU",
            messages=[
                {"role": "system", "content": "You are an expert in wine classification."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.5
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"predicted_country": response.choices[0].message.content.strip()})
        }

    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON format in request body"})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
