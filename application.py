from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys
import pandas as pd
import os
import pickle
import joblib

def add_description_length(df):
    df = df.copy()  # Ensure we don’t modify original data
    df["description_length"] = df["description"].str.len()  # Compute text length
    return df

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

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
            model = joblib.load(model_path)
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
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")
