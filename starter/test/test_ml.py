
import pickle
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import sys
sys.path.insert(0, "./starter/starter/ml")
from data import process_data
from model import train_model, compute_model_metrics, inference


@pytest.fixture(scope="module")
def X_y_train():
    data = pd.read_csv("starter/data/census.csv")
    encoder = pickle.load(open("starter/model/encoder.pickle", "rb"))
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=encoder,
    )
    return X_train, y_train


@pytest.fixture(scope="module")
def model():
    model = pickle.load(open("starter/model/model.pickle", "rb"))
    return model


def test_train_model(X_y_train):
    X_train, y_train = X_y_train
    model = train_model(X_train, y_train)
    assert isinstance(model, sklearn.ensemble.RandomForestClassifier)


def test_compute_model_metrics(X_y_train, model):
    X_train, y_train = X_y_train

    pred = inference(model, X_train)
    metrics = compute_model_metrics(y_train, pred)
    precision, recall, fbeta = metrics

    assert isinstance(precision, np.float64)
    assert isinstance(recall, np.float64)
    assert isinstance(fbeta, np.float64)


def test_inference(X_y_train, model):
    X_train, y_train = X_y_train
    pred = inference(model, X_train)

    assert isinstance(pred, np.ndarray)
