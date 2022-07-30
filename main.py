""" 
Module for web app API
"""
# from typing import Literal
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference

# Alias Generator funtion for class CensusData


def replace_hyphens(string: str) -> str:
    return string.replace("_", "-")


class CustomerData(BaseModel):
    """
    Class to ingest the body from POST
    """

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = replace_hyphens
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


# Instantiate the app
app = FastAPI()


with open("./model/trained_model.pkl", "rb") as f:
    trained_model = pickle.load(f)
with open("./model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("./model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

# Define a GET on the specified andpoint


@app.get("/")
async def get_items():
    """Simple GET"""
    return {"greeting": "Hello World!"}


@app.post("/prediction")
async def make_inference(data: CustomerData):
    """POST to return prediction from our saved model"""
    input_data = data.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False,
    )

    # prediction = trained_model.predict(X)
    prediction = inference(trained_model, X)
    y = lb.inverse_transform(prediction)[0]
    return {"prediction": y}
