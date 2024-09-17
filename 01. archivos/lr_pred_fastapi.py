
from fastapi import FastAPI
from transformers import pipeline
import uvicorn
from pyngrok import ngrok
import pickle
import numpy as np

app = FastAPI()

# Cargar el modelo guardado
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.get("/") #Mensaje al iniciar el aplicativo.
def read_root():
    return {"message": "Welcome to the FastAPI for breast cancer prediction - RA"}

@app.get("/breast_cancer_tab_pred") # Para predicciones con datos tabulares.
def bc_tab_pred(radius_mean: float, texture_mean: float, perimeter_mean: float,
                      area_mean: float, smoothness_mean: float, compactness_mean: float,
                      concavity_mean: float):
  
  # Crear un registro completo con los valores medios para los atributos faltantes
  input_values = np.array([
      radius_mean,
      texture_mean,
      perimeter_mean,
      area_mean,
      smoothness_mean,
      compactness_mean,
      concavity_mean,
      *mean_values[7:]  # Suponiendo que el usuario solo introduce los primeros 7 atributos
  ]).reshape(1, -1)

  # Realizar la predicción utilizando el modelo cargado
  prediction = model.predict(input_values)

  # Devolver la predicción
  return {"breast cancer prediction": int(prediction[0])}
