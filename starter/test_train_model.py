import pytest
import pandas as pd
# from starter.ml.model import *
# from starter.ml.data import process_data
import joblib
import sys
import os
from pathlib import Path

# model_path = Path(__file__).parent.absolute().parents[0].joinpath('model')

root_path = Path(__file__).parent.absolute().parents[0]

# print(root_path.joinpath("starter/slice_model_output.txt"))


@pytest.fixture
def data():
    data = pd.read_csv(root_path.joinpath("data/census.csv"), sep=",", skipinitialspace=True)
    return data


def test_data_size(data):
    """
    Test if we have reasonable amount of data
    """
    assert 1000 < data.shape[0] < 1000000


def test_model():
    """
    Checks that model is saved
    """
    assert os.path.isfile(root_path.joinpath("model/trained_model.pkl"))


def test_encoder():
    """
    Checks that encoder is saved
    """
    assert os.path.isfile(root_path.joinpath("model/encoder.pkl"))


def test_lb():
    """
    Checks lb is saved
    """
    assert os.path.isfile(root_path.joinpath("model/lb.pkl"))


def test_slices():
    """
    Checks that slices functions outputs the scoring slices in the .txt file
    """
    assert os.path.isfile(root_path.joinpath("starter/slice_model_output.txt"))
