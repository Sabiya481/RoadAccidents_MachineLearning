from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED 15-FEATURE MODEL
# ============================================================

model = joblib.load(
    "accident_severity_model_15_features.pkl"
)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Road Accident Severity Prediction API",
    description="ML API for Road Accident Severity Prediction using 15 features",
    version="1.0"
)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class AccidentData(BaseModel):

    Time: str

    Day_of_week: str

    Age_band_of_driver: str

    Sex_of_driver: str

    Driving_experience: str

    Type_of_vehicle: str

    Area_accident_occured: str

    Types_of_Junction: str

    Road_surface_type: str

    Road_surface_conditions: str

    Light_conditions: str

    Weather_conditions: str

    Type_of_collision: str

    Number_of_vehicles_involved: int

    Number_of_casualties: int


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Road Accident Severity Prediction API",
        "status": "API is running successfully",
        "features": 15
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_accident(data: AccidentData):

    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

    input_data = {

        "Time": data.Time,

        "Day_of_week": data.Day_of_week,

        "Age_band_of_driver": data.Age_band_of_driver,

        "Sex_of_driver": data.Sex_of_driver,

        "Driving_experience": data.Driving_experience,

        "Type_of_vehicle": data.Type_of_vehicle,

        "Area_accident_occured":
            data.Area_accident_occured,

        "Types_of_Junction":
            data.Types_of_Junction,

        "Road_surface_type":
            data.Road_surface_type,

        "Road_surface_conditions":
            data.Road_surface_conditions,

        "Light_conditions":
            data.Light_conditions,

        "Weather_conditions":
            data.Weather_conditions,

        "Type_of_collision":
            data.Type_of_collision,

        "Number_of_vehicles_involved":
            data.Number_of_vehicles_involved,

        "Number_of_casualties":
            data.Number_of_casualties
    }


    # --------------------------------------------------------
    # Convert input dictionary to DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(input_df)[0]


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": str(prediction)
    }
