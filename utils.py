import pickle
import json
import os
from Cnn import CustomNeuralNetwork
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_symptoms():
    """Load the list of symptoms from a JSON file."""
    with open('ressources\symptoms_list.json', 'r') as file:
        symptoms_list = json.load(file)
    return symptoms_list


def load_diseases():
    """Load the dictionary of diseases from a JSON file."""
    with open('ressources\disease_dict.json', 'r') as file:
        diseases = json.load(file)
    return diseases


def load_models():
    """Load the trained models"""

    models_directory = os.path.join(os.getcwd(), 'models')
    
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.pkl')]
    models = []
    model_names = []

    for model_file in model_files:
        if model_file[:3] == "cnn":
            continue
        model_path = os.path.join(models_directory, model_file)
        with open(model_path, 'rb') as f:  
            models.append(pickle.load(f))  
        model_names.append(model_file[:-4])  

    return models, model_names

def encode(df):
    """
    Encodes the diagnostic labels into numeric format for model training.
    
    Parameters:
    - df: DataFrame, input dataset with categorical labels.
    
    Returns:
    - df: DataFrame, dataset with encoded labels.
    - encoder: LabelEncoder, fitted encoder for future reference.
    """
    encoder = LabelEncoder()
    df["diagnostic"] = encoder.fit_transform(df["diagnostic"]) 
    return df, encoder


def transfer(df, encoder):
    """
    Saves the encoded disease labels and symptom list to JSON files for later reference.
    
    Parameters:
    - df: DataFrame, dataset with encoded labels.
    - encoder: LabelEncoder, encoder used to transform diagnostic labels.
    """
    disease_dict = {str(idx): name for idx, name in enumerate(encoder.classes_)}
    with open('ressources\disease_dict.json', 'w') as file:
        json.dump(disease_dict, file)  

    symptom_list = df.columns.tolist()[:-1]  
    with open('ressources\symptoms_list.json', 'w') as file:
        json.dump(symptom_list, file) 
        
        
def read(path):
    """
    Reads the dataset from a CSV file, drops columns with missing values,
    balances the dataset, encodes the diagnostic labels, and saves metadata.
    
    Parameters:
    - path: str, path to the dataset CSV file.
    
    Returns:
    - dfencoded: DataFrame, encoded and balanced dataset.
    """
    data = pd.read_csv(path).dropna(axis=1)
    
    datadf = pd.DataFrame(data)
    
    dfencoded, encoder = encode(datadf)

    transfer(dfencoded, encoder)
        
    return dfencoded

def save_model(model, model_name):
    """
    Saves the trained model to a file using pickle.
    
    Parameters:
    - model: Trained model to save.
    - model_name: str, name to use for the saved model file.
    """
    
    models_directory = os.path.join(os.getcwd(), 'models')

    os.makedirs(models_directory, exist_ok=True)

    model_path = os.path.join(models_directory, f'{model_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    

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



def load_cnn(mode ='xavier', epochs = 2500,learning_rate = 0.05,h1=192,h2=192,h3=64):  
    with open(f'models\cnn-{mode}-{epochs}-{learning_rate}-{h1}-{h2}-{h3}.pkl', 'rb') as f:
        params = pickle.load(f)
        model = CustomNeuralNetwork(h1=h1,h2=h2,h3=h3)
        model.W1, model.b1 = params['W1'], params['b1']
        model.W2, model.b2 = params['W2'], params['b2']
        model.W3, model.b3 = params['W3'], params['b3']
        model.W_output, model.b_output = params['W_output'], params['b_output']
    return model,f"cnn-{mode}-{epochs}-{learning_rate}-{h1}-{h2}-{h3}"


def delete_models():
    """Delete all model files in the specified directory.
    
    """
    models_directory = os.path.join(os.getcwd(), 'models')
    
    if os.path.exists(models_directory):
        for filename in os.listdir(models_directory):
            file_path = os.path.join(models_directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  
   
    