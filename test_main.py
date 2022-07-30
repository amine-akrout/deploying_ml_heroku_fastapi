"""
This is unit test for rest_api.py
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    """
    Tests GET. Status code and if it is returning what is expected
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}


def test_post_more_than_50(client):
    """
    Tests POST for a prediction less than 50k.
    Status code and if the prediction is the expected one
    """
    response = client.post(
        "/prediction",
        json={
            "age": 65,
            "workclass": "State-gov",
            "fnlgt": 209280,
            "education": "Masters",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 7000,
            "capital_loss": 0,
            "hours_per_week": 35,
            "native_country": "United-States",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_post_less_than_50(client):
    """
    Tests POST for a prediction more than 50k.
    Status code and if the prediction is the expected one
    """
    response = client.post(
        "/prediction",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
