import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QRadioButton, QButtonGroup, QLineEdit
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt 


def load_symptoms():
    with open('symptoms_list.json', 'r') as file:
        symptoms_list = json.load(file)
    return symptoms_list

def setX(symptoms):
    symptomlist = load_symptoms()
    X = [0] * len(symptomlist)
    for symptom in symptoms:
        if symptom in symptomlist:
            X[symptomlist.index(symptom)] = 1
    return X

def predict(input, models):
    predictions = []
    probabilities = []  # List to store probability outputs
    for model in models:
        reshaped = [input]  
        prediction = model.predict(reshaped)  
        predictions.append(prediction[0]) 
        # Get probabilities
        prob = model.predict_proba(reshaped)
        probabilities.append(prob[0])  # Store the probabilities for the first (and only) sample
    return predictions, probabilities  # Return both predictions and probabilities

def load_diseases():
    with open('disease_dict.json', 'r') as file:
        diseases = json.load(file)
    return diseases

def load_models():
    models = []
    model_names = ['svm_model.pkl', 'nb_model.pkl', 'rf_model.pkl']
    for model_name in model_names:  
        with open(model_name, 'rb') as model_file:
            models.append(pickle.load(model_file))  # Append loaded model
    return models, model_names




def decode_predictions(encoded_predictions):
    disease = load_diseases()
    decoded_predictions = [disease[str(pred)] for pred in encoded_predictions]  # Convert preds to str for dict lookup
    return decoded_predictions




class App(QWidget):
    def __init__(self):
        super().__init__()
        
if __name__ == "__main__":
    symptoms = ["itching","skin_rash","nodal_skin_eruptions"]
    X = setX(symptoms)
    models,model_names = load_models()
    predictions, probabilities = predict(X, models)
    fin = decode_predictions(predictions)
    for idx, model in enumerate(models):
        model_name = model_names[idx]
        disease = fin[idx]
        probability = max(probabilities[idx])
        print(f"Using model '{model_name}', your predicted disease is '{disease}' with a probability of {probability*100:.2f}%.")