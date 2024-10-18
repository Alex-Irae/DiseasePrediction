import pickle
import json

def load_symptoms():
    """Load the list of symptoms from a JSON file."""
    with open('symptoms_list.json', 'r') as file:
        symptoms_list = json.load(file)
    return symptoms_list

def load_diseases():
    """Load the dictionary of diseases from a JSON file."""
    with open('disease_dict.json', 'r') as file:
        diseases = json.load(file)
    return diseases

def load_models():
    """Load the trained models from pickle files."""
    model_names = ['svm_model.pkl', 'nb_model.pkl', 'rf_model.pkl']
    models = [pickle.load(open(model_name, 'rb')) for model_name in model_names]
    return models, model_names

def setX(symptoms):
    """Convert the list of symptoms into a binary feature vector."""
    symptom_list = load_symptoms()
    X = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    return X

def predict(input_data, models):
    """Predict the disease using the trained models."""
    predictions = []
    probabilities = []
    for model in models:
        reshaped_input = [input_data]
        predictions.append(model.predict(reshaped_input)[0])
        probabilities.append(model.predict_proba(reshaped_input)[0])
    return predictions, probabilities

def decode_predictions(encoded_predictions):
    """Decode the encoded predictions into disease names."""
    disease_dict = load_diseases()
    return [disease_dict[str(pred)] for pred in encoded_predictions]