# 🚀 AWS Lambda Wine Country Predictor

This project sets up an **AWS Lambda function** that predicts the country of origin for a wine based on its characteristics. It utilizes **OpenAI's fine-tuned GPT-4 model** and integrates **AWS Secrets Manager** for securely storing the API key.

## 📌 Key Features
- **Lambda Deployment with GitHub Actions**: The function is updated automatically when changes are pushed to the `master` branch.
- **OpenAI GPT-4 Fine-Tuned Model**: Uses a custom-trained model to predict wine origin based on given features.

## 🔧 How It Works
1. The **Lambda function** extracts wine characteristics from the incoming API request.
2. It sends a **formatted prompt** to OpenAI's fine-tuned model for prediction.
3. The **predicted country** is returned as a JSON response.

## 🛠️ Deployment Process
- **GitHub Actions** automatically zips the `lambda_function.py` and updates the AWS Lambda function when changes are pushed.
- AWS CLI is used to upload the new function code.
