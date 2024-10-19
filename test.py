import os
from disease_predict import load_models,load_symptoms

x = load_symptoms() 
print(x)
print("----------------------------------------------------")
print(x[2])