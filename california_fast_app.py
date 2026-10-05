from fastapi import FastAPI
import numpy as np
import joblib
import uvicorn
#-----------------------------------
app=FastAPI()
#------------------------------------
obj= joblib.load('california.joblib')
model=obj['model']
cols=obj['columns']
#------------------------------------
@app.get('/')
def main():
    return('welcome')

@app.get('/predict')
def predict(MedInc:float,HouseAge:float,AveRooms:float,AveBedrms:float,Population:float,AveOccup:float,Longitude:float,Latitude:float):
    Input=[[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Longitude,Latitude]]
    Out=model.predict(Input)
    return(f'The med House val is:{Out}')

if __name__=='__main__':
    uvicorn.run(app,host="0.0.0.0", port=8000)    