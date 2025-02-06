from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import GridSearchCV
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
nltk.download("punkt")  # Required for word_tokenize
nltk.download("stopwords")  # Required for stopwords
nltk.download("wordnet")  # Required for Lemmatization
import joblib

def add_description_length(df):
    df = df.copy()  # Ensure we don’t modify original data
    df["description_length"] = df["description"].str.len()  # Compute text length
    return df

# custom_stopwords = set([
#     "wine", "flavor", "taste", "aroma",  # Example wine-related words
#     "bottle", "vintage", "palate"       # You can add more words here
# ])

# def clean (text):
#     for punctuation in string.punctuation:
#         text = text.replace(punctuation, ' ') # Remove Punctuation
#     lowercased = text.lower() # Lower Case
#     tokenized = word_tokenize(lowercased) # Tokenize
#     words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
#     stop_words = set(stopwords.words('english')) # Make stopword list
#     stop_words.update(custom_stopwords)
#     without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
#     lemma=WordNetLemmatizer() # Initiate Lemmatizer
#     lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
#     cleaned = ' '.join(lemmatized) # Join back to a string
#     return cleaned

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,data_scaled):
        try:
            model_path=os.path.join("artifacts","best_model.pkl")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            print(f"model_path: {model_path}")
            # model=load_object(file_path=model_path)
            model = joblib.load(model_path)
            print(type(model))
            # preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            print(data_scaled)
            # data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        description: str,
        points: int,
        price: int,
        variety: str):

        self.description = description
        self.points = points
        self.price = price
        self.variety = variety

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "description": [self.description],
                "points": [self.points],
                "price": [self.price],
                "variety": [self.variety]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            description=request.form.get('Description'),
            points=int(request.form.get('Points')),
            price=int(request.form.get('Price')),
            variety=request.form.get('Variety')

        )
        pred_df=data.get_data_as_data_frame()
        print("LATEST:\n", pred_df.to_string(index=False))
        print("Before Prediction.....cleaning")

        # pred_df['description_clean'] = pred_df.description.apply(clean)
        print("LATEST2:\n", pred_df.to_string(index=False))
        # print("Before Prediction.....cleaning done")
        # pred_df["price_log"] = np.log1p(pred_df["price"])
        # print("LATEST3:\n", pred_df.to_string(index=False))
        # pred_df["text_length"] = pred_df["description_clean"].apply(lambda x: len(x.split()))
        # print("LATEST4:\n", pred_df.to_string(index=False))
        # print("Before Prediction.....scalling")
        # scaler = StandardScaler()
        # pred_df[["points_scaled", "price_scaled","length_scaled"]] = scaler.fit_transform(pred_df[["points", "price_log","text_length"]])
        # print("LATEST5:\n", pred_df.to_string(index=False))
        # print("Before Prediction.....scalling DONE")
        # label_encoder = LabelEncoder()
        # pred_df["variety_encoded"] = label_encoder.fit_transform(pred_df["variety"])
        # print("Before Prediction.....ENCODING DONE")
        # df_transformed = pred_df[["points_scaled","description_clean","variety_encoded","length_scaled","price_scaled"]]
        # print(f"Here goes modelling with {df_transformed}")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")
