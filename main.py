from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = FastAPI()

model = joblib.load('ElasticNet_model.pkl')
scaler = joblib.load('scaler.pkl')
mis_replacer = joblib.load('imputer.pkl')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class CarItemList(BaseModel):
    items: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    input_data = pd.DataFrame([item.dict()])

    input_data[['torque', 'max_torque_rpm']] = input_data['torque'].str.extract('(\d+\.\d+)Nm@ (\d+)rpm').astype(float)
    input_data['mileage'] = pd.to_numeric(input_data['mileage'], errors='coerce').astype(float)
    input_data['engine'] = pd.to_numeric(input_data['engine'], errors='coerce').astype(float)
    input_data['max_power'] = pd.to_numeric(input_data['max_power'], errors='coerce').astype(float)

    cat_features_mask = (input_data.dtypes == "object").values
    input_real = input_data[input_data.columns[~cat_features_mask]]
    input_cat = input_data[input_data.columns[cat_features_mask]].fillna("")

    input_data = pd.concat([input_cat, pd.DataFrame(mis_replacer.transform(input_real), columns=input_real.columns)],
                           axis=1)

    input_data['engine'] = input_data['engine'].astype(int)
    input_data['seats'] = input_data['seats'].astype(int)
    input_data = input_data.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner'], axis=1)

    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
    input_data_scaled = input_data_scaled.drop(['selling_price', 'torque', 'engine'], axis=1)

    prediction = model.predict(input_data_scaled)
    return prediction[0]

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> CarItemList:
    try:
        df = pd.read_csv(BytesIO(await file.read()))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")

    df[['torque', 'max_torque_rpm']] = df['torque'].str.extract('(\d+\.\d+)Nm@ (\d+)rpm').astype(float)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce').astype(float)
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce').astype(float)
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce').astype(float)

    cat_features_mask = (df.dtypes == "object").values
    df_real = df[df.columns[~cat_features_mask]]
    df_cat = df[df.columns[cat_features_mask]].fillna("")

    df = pd.concat([df_cat, pd.DataFrame(mis_replacer.transform(df_real), columns=df_real.columns)], axis=1)

    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df = df.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner'], axis=1)

    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    df_scaled = df_scaled.drop(['selling_price', 'torque', 'engine'], axis=1)

    predictions = model.predict(df_scaled)
    df['predictions'] = predictions

    csv_data = df.to_csv(index=False)
    return StreamingResponse(BytesIO(csv_data.encode()), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})

