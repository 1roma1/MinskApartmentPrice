import joblib
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class ApartmentDesc(BaseModel):
    year_built: int
    floor: int 
    house_type: str 
    ceiling_height: float 
    total_area: int 
    living_area: int 
    kitchen_area: int 
    bathroom: str 
    balcony: str 
    district: str
    subway: int 
    number_of_rooms: int 
    number_of_storeys: int 


app = FastAPI()
model = joblib.load("model.pk")


@app.post("/predict")
def predict(data: ApartmentDesc):
    data = dict(data)
    df = pd.DataFrame.from_dict(data, orient='index').transpose()      
    price = np.expm1(model.predict(df))
    return {"price": price[0]}
