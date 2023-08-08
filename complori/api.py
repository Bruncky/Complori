import pandas as pd

# IMPORT FUNCTION TO LOAD MODEL FROM STORAGE HERE

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()

@app.get("/predict")
def predict(
        # Params to describe a given apartment as see in the dataset
    ):

    # Here is the logic to get the data and feed it to the loaded model
    model = app.state.model
    assert model is not None

    # The pipeline already takes care of getting the data and preprocessing it
    X_processed = fit_pipeline()
    y_pred = model.predict(X_processed)

    return dict(sale_price = float(y_pred))


@app.get("/")
def root():
    # Dummy to know the API works
    return dict(greeting = 'Hello')
