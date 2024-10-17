import pickle
import json

def load_symptoms():
    with open('symptoms_list.json', 'r') as file:
        symptoms_list = json.load(file)
    return symptoms_list


def load_diseases():
    with open('disease_dict.json', 'r') as file:
        diseases = json.load(file)
    return diseases

def load_models():
    models = []
    model_names = ['svm_model.pkl', 'nb_model.pkl', 'rf_model.pkl']
    for model_name in model_names:  
        with open(model_name, 'rb') as model_file:
            models.append(pickle.load(model_file))  
    return models, model_names


def setX(symptoms):
    symptomlist = load_symptoms()
    X = [0] * len(symptomlist)
    for symptom in symptoms:
        if symptom in symptomlist:
            X[symptomlist.index(symptom)] = 1
    return X

def predict(input, models):
    predictions = []
    probabilities = []  
    for model in models:
        reshaped = [input]  
        
        prediction = model.predict(reshaped)  
        predictions.append(prediction[0]) 
        
        prob = model.predict_proba(reshaped)
        probabilities.append(prob[0])  
    return predictions, probabilities  


def decode_predictions(encoded_predictions):
    disease = load_diseases()
    decoded_predictions = [disease[str(pred)] for pred in encoded_predictions]  # Convert preds to str for dict lookup
    return decoded_predictions
