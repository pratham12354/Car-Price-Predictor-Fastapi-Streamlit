from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import pandas as pd
import numpy as np
import pickle
import os
from enum import Enum

app = FastAPI()

# Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "car_price_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class CompanyEnum(str, Enum):
    Hyundai = "Hyundai"
    Mahindra = "Mahindra"
    Ford = "Ford"
    Maruti = "Maruti"
    Skoda = "Skoda"
    Audi = "Audi"
    Toyota = "Toyota"
    Renault = "Renault"
    Honda = "Honda"
    Datsun = "Datsun"
    Mitsubishi = "Mitsubishi"
    Tata = "Tata"
    Volkswagen = "Volkswagen"
    Chevrolet = "Chevrolet"
    Mini = "Mini"
    BMW = "BMW"
    Nissan = "Nissan"
    Hindustan = "Hindustan"
    Fiat = "Fiat"
    Force = "Force"
    Mercedes = "Mercedes"
    Land = "Land"
    Jaguar = "Jaguar"
    Jeep = "Jeep"
    Volvo = "Volvo"


class userInput(BaseModel):
    year: Annotated[int, Field(..., gt=1994, lt=2020)]
    km_driven: Annotated[int, Field(..., gt=-1, lt=500000)]
    fuel_type: Annotated[Literal["Petrol", "Diesel", "LPG"], Field(...)]
    company: Annotated[CompanyEnum, Field(...)]
    car_name: Annotated[str, Field(...)]


@app.post("/predict")
def predict_car_price(data: userInput):
    try:
        input_df = pd.DataFrame([{
            "name": data.car_name,
            "company": data.company,
            "year": data.year,
            "kms_driven": data.km_driven,
            "fuel_type": data.fuel_type
        }])

        predicted_log_price = model.predict(input_df)[0]
        predicted_price = float(np.exp(predicted_log_price))

        return {"predicted_price": predicted_price}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
