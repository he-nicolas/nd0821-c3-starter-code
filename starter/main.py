import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data

# Instantiate the app.
app = FastAPI()

# Load the model
model = pickle.load(open("starter/model/model.pickle", "rb"))
encoder = pickle.load(open("starter/model/encoder.pickle", "rb"))
lb = pickle.load(open("starter/model/lb.pickle", "rb"))
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


class dataInput(BaseModel):
    age: int = Field(example=45)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13, alias="education-num")
    marital_status: str = Field(
        example="Never-married", 
        alias="marital-status")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=2174, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=40, alias="hours-per-week")
    native_country: str = Field(
        example="United-States", 
        alias="native-country")


# Define a GET on the specified endpoint.
@app.get("/")
async def welcome_message():
    return {"greeting": "Hello Model User!"}


# Use POST action to send data to the server
@app.post("/inference")
async def model_inference(input_data: dataInput):
    data_dict = input_data.dict(by_alias=True)
    data_df = pd.DataFrame(data_dict, index=[0])

    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = model.predict(X).tolist()
    return {"result": prediction[0]}
