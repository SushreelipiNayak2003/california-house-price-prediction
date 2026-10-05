from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import joblib
import uvicorn

# -----------------------------------
app = FastAPI()

# -----------------------------------
# Static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Templates
templates = Jinja2Templates(directory="templates")

# -----------------------------------
# Load model
obj = joblib.load('california.joblib')

model = obj['model']
cols = obj['columns']

# -----------------------------------

@app.get('/')
def main(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# -----------------------------------

@app.get('/predict')
def predict(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Longitude: float,
    Latitude: float
):

    Input = [[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Longitude,
        Latitude
    ]]

    Out = model.predict(Input)

    return {
        "prediction": float(Out[0])
    }


# -----------------------------------

if __name__ == '__main__':

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )