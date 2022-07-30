""" 
Module for web app API
"""
import os

# from typing import Literal
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from starter.ml.data import process_data
from starter.ml.model import inference


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


# Instantiate the app
app = FastAPI()

# Define a GET on the specified andpoint


@app.get("/")
async def get_items():
    """Simple GET"""
    return {"greeting": "Hello World!"}


@app.post("/prediction")
async def make_inference(data: CustomerData):
    """POST to return prediction from our saved model"""
    with open("./model/trained_model.pkl", "rb") as f:
        trained_model = pickle.load(f)
    with open("./model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("./model/lb.pkl", "rb") as f:
        lb = pickle.load(f)

    input_data = data.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
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
