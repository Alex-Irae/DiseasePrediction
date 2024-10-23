import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier  
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE  
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import tensorflow as tf
from utils import read, save_model


filepath = os.getcwd() + "\\ressources\data.csv"



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
    
    scaler = MinMaxScaler()  
    X_scaled = scaler.fit_transform(X_smote)  
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  
    
    df = pd.concat([X_scaled_df, pd.DataFrame(y_smote, columns=["diagnostic"])], axis=1)
    
    return df



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

    X_noisy = add_noise(X_train, noise_level=0.05)
    
    
    Xus = df_unseen.iloc[:, :-1]  
    Yus = df_unseen.iloc[:, -1]   

    return X_noisy, Y_train, Xus, Yus




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
    with tf.device('/GPU:0'):
        # SVM hyperparameter tuning
        svm_pipeline = Pipeline([
            ('rfe', RFE(estimator=SVC(kernel='linear'), n_features_to_select=66)),  # RFE for feature selection
            ('poly', PolynomialFeatures(degree=1, interaction_only=True)),          # Polynomial features
            ('svm', SVC(probability=True))                                          # SVM classifier
        ])
        
        svm_params = {
            'svm__C': [0.1, 1, 10],               # Regularization strength
            'svm__kernel': ['linear', 'rbf'],  # Type of kernel to use
            'svm__gamma': [0.01, 0.1, 1]         # Kernel coefficient for 'rbf'
        }
        
        svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(X_train, Y_train)
        print(f"Best SVM parameters: {svm_grid.best_params_}")  
        print(f"Best SVM accuracy: {svm_grid.best_score_}") 
        
        print("rf")
        # Random Forest hyperparameter tuning
        rf_pipeline = Pipeline([
            ('rfe', RFE(estimator=RandomForestClassifier(random_state=18), n_features_to_select=66)),  # RFE for feature selection
            ('rf', RandomForestClassifier(random_state=18))                                          # RF classifier
        ])
        
        rf_params = {
            'rf__n_estimators': [100, 200, 300],  # Number of trees in the forest
            'rf__max_depth': [10, 20, 30, None],  # Maximum depth of the tree
            'rf__min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
            'rf__min_samples_leaf': [1, 2]  # Minimum number of samples required at a leaf node
        }
        
        rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='accuracy', n_jobs=-1)  
        rf_grid.fit(X_train, Y_train)  
        print(f"Best RF parameters: {rf_grid.best_params_}")  
        print(f"Best RF accuracy: {rf_grid.best_score_}") 
        
        print("lr")
        # Logistic Regression hyperparameter tuning
        lr_pipeline = Pipeline([
            ('rfe', RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=66)),  # RFE for feature selection
            ('lr', LogisticRegression(max_iter=1000))                                          
        ])
        
        lr_params = {
            'lr__C': [0.01, 0.1, 1, 10],  # Regularization strength
            'lr__solver': ['liblinear', 'lbfgs'],  # Different solvers
            'lr__penalty': ['l2']  # L2 regularization
        }
        
        lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring='accuracy', n_jobs=-1)
        lr_grid.fit(X_train, Y_train)
        print(f"Best LR parameters: {lr_grid.best_params_}")  
        print(f"Best LR accuracy: {lr_grid.best_score_}") 
    
        print("xgb")	
        # XGBoost hyperparameter tuning
        xgb_pipeline = Pipeline([
            ('rfe', RFE(estimator=XGBClassifier(eval_metric='mlogloss'), n_features_to_select=66)),  # RFE for feature selection
            ('xgb', XGBClassifier(eval_metric='mlogloss'))                                          
        ])
        
        xgb_params = {
            'xgb__n_estimators': [50, 100, 200],  # Number of boosting rounds
            'xgb__learning_rate': [0.01, 0.1, 0.2],  # Learning rate
            'xgb__max_depth': [3, 5, 7],  # Maximum depth of the trees
            'xgb__subsample': [0.8, 1.0],  # Fraction of samples to be used for fitting the individual base learners
            'xgb__colsample_bytree': [0.8, 1.0]  # Fraction of features to be randomly sampled for each tree
        }
        
        xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
        xgb_grid.fit(X_train, Y_train)
        print(f"Best XGB parameters: {xgb_grid.best_params_}")  
        print(f"Best XGB accuracy: {xgb_grid.best_score_}") 
        
    return svm_grid.best_estimator_, rf_grid.best_estimator_, lr_grid.best_estimator_, xgb_grid.best_estimator_



def add_noise(X, noise_level=0.01):
    """
    Adds noise to binary features by flipping a small percentage of values.
    
    Parameters:
    - X: numpy array, original feature data (binary 0/1).
    - noise_level: float, percentage of data points to flip (default is 1%).
    
    Returns:
    - X_noisy: numpy array, feature data with noise added.
    """
    X_noisy = X.copy()  # Create a copy of the DataFrame
    
    # Number of elements to modify
    num_elements = int(noise_level * X.size)
    
    # Randomly select indices to flip
    row_indices = np.random.randint(0, X.shape[0], num_elements)
    col_indices = np.random.randint(0, X.shape[1], num_elements)
    
    # Flip values at selected indices using iloc in a loop
    for row, col in zip(row_indices, col_indices):
        X_noisy.iloc[row, col] = 1 - X_noisy.iloc[row, col]  # Flip between 0 and 1
    
    return X_noisy

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

    # Initialize models and perform hyperparameter tuning
    # consider swtiching to GPU for computaion
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
     
    with tf.device('/GPU:0'):
        voting_clf.fit(X_train, Y_train)
        cv_score = cross_val_score(voting_clf, X_train, Y_train, cv=cv, scoring='accuracy').mean()
    print(f"voting_classifier Scores: {cv_score} | Mean Accuracy: {cv_score.mean()}")
    
    # save_model(voting_clf, 'voting_classifier')
    
    # for model, model_name in zip(models, model_names):
         
        # save_model(model, model_name)  
     
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
    launch(df)  # Start the training process