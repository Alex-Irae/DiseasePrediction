import lime
import numpy as np
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier  
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE  
import pickle
import json
from disease_predict import load_models
import matplotlib.pyplot as plt



filepath = os.getcwd() + "\\ressources\data.csv"




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
    
    df = balance(datadf)

    dfencoded, encoder = encode(df)

    transfer(dfencoded, encoder)
    
    return dfencoded




def balance(df):
    """
    Balances the dataset using SMOTE to oversample minority classes.
    
    Parameters:
    - df: DataFrame, input dataset with potential class imbalance.
    
    Returns:
    - df: DataFrame, balanced dataset.
    """
    # use Smote to balance the data
    smote = SMOTE(random_state=42)
    X = df.iloc[:, :-1]  
    y = df['diagnostic'] 
    X_smote, y_smote = smote.fit_resample(X, y)  
    df = pd.concat([X_smote, pd.DataFrame(y_smote, columns=["diagnostic"])], axis=1)  
    
    return df




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
    df["diagnostic"] = encoder.fit_transform(df["diagnostic"]) # encode the labels (from name to nnumber)
    return df, encoder





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
    




def delete_models():
    """Delete all model files in the specified directory.
    
    """
    models_directory = os.path.join(os.getcwd(), 'models')
    
    if os.path.exists(models_directory):
        for filename in os.listdir(models_directory):
            file_path = os.path.join(models_directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  
   
    
    

def categorize(df):
    """
    Splits the dataset into training and unseen (test) data.
    
    Parameters:
    - df: DataFrame, input dataset grouped by diagnostic labels.
    
    Returns:
    - X_train: DataFrame, features for training.
    - Y_train: Series, labels for training.
    - Xus: DataFrame, features for unseen (test) data.
    - Yus: Series, labels for unseen (test) data.
    """
    
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




def hyperparameter_tuning(X_train, Y_train):
    """
    Tunes hyperparameters for SVM, Random Forest, LogisticRegression and XGBoost models using GridSearchCV.
    
    Parameters:
    - X_train: DataFrame, features for training.
    - Y_train: Series, labels for training.
    
    Returns:
    - svm_model: Trained SVM model with best parameters.
    - rf_model: Trained Random Forest model with best parameters.
    - lr_model : Trained logistic regression model with best parameters.
    - xgb_model: Trained XGBoost model with best parameters.
    """
    
    # SVM hyperparameter tuning
    svm_params = {
        'C': [0.1, 1, 10],               # Regularization strength
        'kernel': ['linear', 'rbf'],  # Type of kernel to use
        'gamma': [0.01, 0.1, 1]         # Kernel coefficient for 'rbf'
    }
    svm_model = SVC(probability=True)  
    svm_grid = GridSearchCV(svm_model, svm_params, cv=5, scoring='accuracy', n_jobs=-1)  
    svm_grid.fit(X_train, Y_train)   
    print(f"Best SVM parameters: {svm_grid.best_params_}")  
    print(f"Best SVM accuracy: {svm_grid.best_score_}") 
    
    
    # RandomForest hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
        'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2]  # Minimum number of samples required at a leaf node
    }
    rf_model = RandomForestClassifier(random_state=18)  
    rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1)  
    rf_grid.fit(X_train, Y_train)  
    print(f"Best RF parameters: {rf_grid.best_params_}")  
    print(f"Best RF accuracy: {rf_grid.best_score_}") 
    
    # Logistic Regression hyperparameter tuning
    lr_params = {
        'C': [0.01, 0.1, 1, 10],  # Regularization strength
        'solver': ['liblinear', 'lbfgs'],  # Different solvers
        'penalty': ['l2']  # L2 regularization
    }
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_grid = GridSearchCV(lr_model, lr_params, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, Y_train)
    print(f"Best LR parameters: {lr_grid.best_params_}")  
    print(f"Best LR accuracy: {lr_grid.best_score_}") 
   
   # XGBoost hyperparameter tuning
    xgb_params = {
        'n_estimators': [50, 100, 200],  # Number of boosting rounds
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'max_depth': [3, 5, 7],  # Maximum depth of the trees
        'subsample': [0.8, 1.0],  # Fraction of samples to be used for fitting the individual base learners
        'colsample_bytree': [0.8, 1.0]  # Fraction of features to be randomly sampled for each tree
    }
    
    xgb_model = XGBClassifier(eval_metric='mlogloss')
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_train, Y_train)
    print(f"Best XGB parameters: {xgb_grid.best_params_}")  
    print(f"Best XGB accuracy: {xgb_grid.best_score_}") 
    
    return svm_grid.best_estimator_, rf_grid.best_estimator_,lr_grid.best_estimator_,xgb_grid.best_estimator_




def train(X_train, Y_train, Xus, Yus):
    """
    Trains multiple models and saves them for future use.
    
    Parameters:
    - X_train: DataFrame, features for training.
    - Y_train: Series, labels for training.
    - Xus: DataFrame, features for unseen (test) data.
    - Yus: Series, labels for unseen (test) data.
    """
    cv = 15
    treshold = 0.95
    
    delete_models()  # Delete existing model files
    
    # Initialize models and perform hyperparameter tuning
    svm_model, rf_model, lr_model, xgb_model  = hyperparameter_tuning(X_train, Y_train) 

    models_list = [svm_model,rf_model,lr_model, xgb_model]  # List of models
    model_names_list = ["svm_model","rf_model","lr_model", "xgb_model"]  

    # list to hold models that meet the performance criterion
    selected_models = []
    selected_models_names = []

    for model,model_name in zip(models_list,model_names_list):
        scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring='accuracy').mean()
        if scores.mean() >= treshold:  
            selected_models.append(model)  
            selected_models_names.append(model_name)
            print(f"{model_name} Scores: {scores} | Mean Accuracy: {scores.mean()}")

    models = selected_models
    model_names = selected_models_names
    
    
    # if no model meets the criteria, use all models
    if not models :
        models = models_list
        model_names = model_names
        print("No model meets the performance criterion. All models will be used.")
        
    # creating a voting classifier with the selected models (soft to get the proba)
    voting_clf = VotingClassifier(estimators=[(name, model) for name, model in zip(model_names, models)],voting='soft')  
        
    voting_clf.fit(X_train, Y_train)
    cv_score = cross_val_score(voting_clf, X_train, Y_train, cv=cv, scoring='accuracy').mean()
    print(f"voting_classifier Scores: {cv_score} | Mean Accuracy: {cv_score.mean()}")
    
    save_model(voting_clf, 'voting_classifier')
    
    for model, model_name in zip(models, model_names):
         
        save_model(model, model_name)  
     
    arrays = [Xus, Yus]  # Unseen data arrays
    prediction(voting_clf, arrays)  # Predict and visualize results





def prediction(voting_clf, arrays):
    """
    Predicts on unseen data using trained models and evaluates the combined model.
    
    Parameters:
    - models: list, list of trained models.
    - arrays: list, contains features and labels for unseen data.
    """
    
    Xus, Yus = arrays  # Unpack unseen data


    voting_clf.fit(Xus, Yus)  
    final_preds = voting_clf.predict(Xus)  

    print(f"Accuracy on Test dataset by the combined model: {accuracy_score(Yus, final_preds) * 100:.2f}%")  

    # Confusion matrix
    cf_matrix = confusion_matrix(Yus, final_preds)  
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)  
    plt.title("Confusion Matrix for Combined Model on Test Dataset")  
    plt.show()   
    
        
        
        
        
        
def launch(df):
    """
    Launches the training process.
    
    Parameters:
    - df: DataFrame, input dataset for training.
    """
    X_train, Y_train, Xus, Yus = categorize(df)  # Split dataset
    train(X_train, Y_train, Xus, Yus)  # Train models
    

if __name__ == "__main__":
    df = read(filepath)  # Read the dataset
    # launch(df)  # Start the training process