
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import uvicorn
from pyngrok import ngrok
import pickle
import numpy as np
import pandas as pd
import joblib

# Cargar el modelo guardado
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar los valores medios de los datos de entrenamiento
mean_values = np.loadtxt('mean_values.txt')

# Cargar los valores de escalado
scaler = joblib.load('scaler.save')

app = FastAPI()

@app.get("/") #Mensaje al iniciar el aplicativo.
def read_root():
    return {"message": "Welcome to the FastAPI for breast cancer prediction - RA"}

@app.get("/breast_cancer_tab_pred_man") # Para predicciones con datos tabulares manualmente
def breast_cancer_tab_pred_man(      smoothness_mean: float, compactness_mean: float,
                      symmetry_mean: float, fractal_dimension_mean: float,
                      texture_se: float, area_se: float,
                      smoothness_se: float, compactness_se: float,
                      concavity_se: float, concave_points_se: float, symmetry_se: float,
                      fractal_dimension_se: float, texture_worst: float,
                      area_worst: float, smoothness_worst: float,
                      compactness_worst: float, concavity_worst: float,
                      concave_points_worst: float, symmetry_worst: float,
                      fractal_dimension_worst: float):

  # Crear un registro completo con los valores medios para los atributos faltantes

  input_values = np.array([
          smoothness_mean,
          compactness_mean,
          symmetry_mean,
          fractal_dimension_mean,
          texture_se,
          area_se,
          smoothness_se,
          compactness_se,
          concavity_se,
          concave_points_se,
          symmetry_se,
          fractal_dimension_se,
          texture_worst,
          area_worst,
          smoothness_worst,
          compactness_worst,
          concavity_worst,
          concave_points_worst,
          symmetry_worst,
          fractal_dimension_worst
      ]).reshape(1, -1)

  input_values = scaler.transform(input_values)

  # Realizar la predicción utilizando el modelo cargado
  prediction = model.predict(input_values)

  # Devolver la predicción
  return {"breast cancer prediction": int(prediction[0])}

@app.post("/load_file_test") # Para hacer test de cargar un archivo '.csv'.
async def load_file(file: UploadFile = File(...)):

  # Read CSV data
  if file.filename.endswith('.csv'):
    data = await file.read()

  # Save loaded CSV file
  with open(file.filename, 'wb') as f:
    f.write(data)

  # Devolver mensaje
  return {"Message": "File loaded successfully"}

@app.get("/breast_cancer_tab_pred_csv") # Para predicciones con datos tabulares desde un fichero csv cargado previamente
def breast_cancer_tab_pred_file( file_name: str):
  
  try: 
    # Cargar el archivo csv
    data = pd.read_csv(file_name, sep=';', decimal='.')

    # Escalar los valores para el modelo de predicción
    input_values = scaler.transform(data)

    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict(input_values)

    # Categorización (label)
    prediction = np.where(prediction == 0, 'benign', 'malignant')

    # Devolver la predicción
    return {"breast cancer prediction": prediction.tolist()}

  except FileNotFoundError:
    return {"Error": "File not found"}
