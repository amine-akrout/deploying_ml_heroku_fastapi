"""
This is unit test for rest_api.py
"""

import pytest
import pickle
from fastapi.testclient import TestClient
from main import app
from pathlib import Path

root_path = Path(__file__).parent.absolute()


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
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 7000,
            "capital-loss": 0,
            "hours-per-week": 35,
            "native-country": "United-States",
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
        },
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
