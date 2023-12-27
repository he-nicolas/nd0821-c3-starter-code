from fastapi.testclient import TestClient
import json
import sys
# Import our app from main.py.
sys.path.insert(0, "./starter")
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello Model User!"}


def test_api_model_inference_0():
    data = json.dumps({
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
        "native-country": "United-States"
    })
    r = client.post("/inference", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {"result": 0}


def test_api_model_inference_1():
    data = json.dumps({
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    })
    r = client.post("/inference", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {"result": 1}
