import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statistics
import pickle
import json

# Filepath to the dataset
filepath = "data.csv"

def read(path):
    # Read the CSV file and drop columns with missing values
    data = pd.read_csv(path).dropna(axis=1)
    datadf = pd.DataFrame(data)
    # Balance the dataset
    df = balance(datadf)
    # Encode the dataset
    dfencoded, encoder = encode(df)
    # Transfer the encoded data and encoder to JSON files
    transfer(dfencoded, encoder)
    return dfencoded

def balance(df):
    # Check the balance of the 'diagnostic' column
    disease_count = df["diagnostic"].value_counts()
    balancemin = disease_count.min()

    # Balance the dataframe by reducing classes with more samples to the minimum count
    if balancemin != disease_count.mean():
        # Identify diseases with more samples than the minimum
        tobalance = disease_count[disease_count > balancemin].index

        for disease in tobalance:
            disease_indices = df[df["diagnostic"] == disease].index
            # Randomly sample to keep the number of samples equal to the minimum
            indices_to_keep = df[df["diagnostic"] == disease].sample(balancemin, random_state=42).index
            indices_to_drop = set(disease_indices) - set(indices_to_keep)
            df = df.drop(indices_to_drop)
    return df

def transfer(df, encoder):
    # Create a dictionary mapping encoded labels to disease names
    disease_dict = {str(idx): name for idx, name in enumerate(encoder.classes_)}
    with open('disease_dict.json', 'w') as file:
        json.dump(disease_dict, file)

    # Save the list of symptoms to a JSON file
    symptom_list = df.columns.tolist()[:-1]
    with open('symptoms_list.json', 'w') as file:
        json.dump(symptom_list, file)

def encode(df):
    # Encode the 'diagnostic' labels into numeric format for model training
    encoder = LabelEncoder()
    df["diagnostic"] = encoder.fit_transform(df["diagnostic"])
    return df, encoder

def save_model(model, model_name):
    # Save the trained model to a file
    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)

def categorize(df):
    # Group the dataframe by 'diagnostic' labels
    df = df.groupby("diagnostic")

    train_ = []  # List to hold training samples
    unseen_ = []  # List to hold unseen (test) samples

    # Split each group into training and unseen data
    for class_label, group in df:
        train, unseen = train_test_split(group, test_size=0.2, random_state=42)  # 80/20 split
        train_.append(train)
        unseen_.append(unseen)

    # Shuffle the training and unseen data
    df_train = pd.concat(train_).sample(frac=1, random_state=42)
    df_unseen = pd.concat(unseen_).sample(frac=1, random_state=42)

    # Prepare feature and label sets for training and unseen data (X features, Y labels)
    X_train = df_train.iloc[:, :-1]
    Y_train = df_train.iloc[:, -1]

    Xus = df_unseen.iloc[:, :-1]
    Yus = df_unseen.iloc[:, -1]

    return X_train, Y_train, Xus, Yus

def train(X_train, Y_train, Xus, Yus):
    # Initialize models
    svm_model = SVC(probability=True)
    nb_model = GaussianNB()
    rf_model = RandomForestClassifier(random_state=18)

    # Train the models on the training data
    svm_model.fit(X_train, Y_train)
    nb_model.fit(X_train, Y_train)
    rf_model.fit(X_train, Y_train)

    models = [svm_model, nb_model, rf_model]
    model_names = ["svm_model", "nb_model", "rf_model"]

    arrays = [Xus, Yus]
    for model, model_name in zip(models, model_names):
        save_model(model, model_name)

    # Uncomment the following line to visualize trained data
    predict(models, arrays)

def predict(models, arrays):
    # Predict on unseen data using the trained models
    svm_model, nb_model, rf_model = models
    Xus, Yus = arrays

    # Predictions on unseen data
    svm_preds = svm_model.predict(Xus)
    nb_preds = nb_model.predict(Xus)
    rf_preds = rf_model.predict(Xus)

    # Combine predictions using mode
    results = [stats.mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

    print(f"Accuracy on Test dataset by the combined model: {accuracy_score(Yus, results) * 100:.2f}%")

    # Confusion matrix
    cf_matrix = confusion_matrix(Yus, results)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title("Confusion Matrix for Combined Model on Test Dataset")
    plt.show()

def score(estimator, X, Y):
    # Calculate accuracy score for cross-validation
    return accuracy_score(Y, estimator.predict(X))

def cvKFold(X, Y, cv):
    # Perform cross-validation using K-Fold
    models = {
        "SVC": SVC(),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=18)
    }
    for model_name, model in models.items():
        scores = cross_val_score(model, X, Y, cv=cv, n_jobs=-1, scoring=score)
        print(model_name)
        print(f"Scores: {scores}")
        print()

def launch(df):
    # Launch the training process
    X_train, Y_train, Xus, Yus = categorize(df)
    train(X_train, Y_train, Xus, Yus)
    # Uncomment to perform cross-validation
    cvKFold(X_train, Y_train, cv=10)

if __name__ == "__main__":
    # Read the dataset and launch the training process
    df = read(filepath)
    launch(df)