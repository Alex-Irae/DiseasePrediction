import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE  
import matplotlib.pyplot as plt
from Cnn import CustomNeuralNetwork
from utils import decode_predictions,load_cnn,read
from sklearn.preprocessing import OneHotEncoder
from utils import setX
import optuna
import tensorflow as tf




def preprocess(df):
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

    for class_label, group in df:
        train, unseen = train_test_split(group, test_size=0.2, random_state=42)  # 80/20 split
        train_.append(train)
        unseen_.append(unseen)

    # Shuffle the training and unseen data
    df_train = pd.concat(train_).sample(frac=1, random_state=42).reset_index(drop=True)
    df_unseen = pd.concat(unseen_).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = df_train.iloc[:, :-1]  
    Y_train = df_train.iloc[:, -1].values.reshape(-1, 1)  
    
    # add noise for better generalisation
    X_noisy = add_noise(X_train, noise_level=0.1)

    encoder = OneHotEncoder()
    Y_train = encoder.fit_transform(Y_train)  
    Xus = df_unseen.iloc[:, :-1]  
    Yus = df_unseen.iloc[:, -1].values.reshape(-1, 1)  
    Yus = encoder.transform(Yus)  

    return X_noisy, Y_train, Xus, Yus



def prediction(clf, arrays):
    """
    Predicts on unseen data using trained models and evaluates the combined model.
    
    Parameters:
    - models: list, list of trained models.
    - arrays: list, contains features and labels for unseen data.
    """
    Xus, Yus = arrays  

    final_preds = clf.predict(Xus)  

    # Convert one-hot encoded Yus back to class labels
    Yus_labels = np.argmax(Yus.toarray(), axis=1)

    # Convert predictions to class labels if needed
    final_preds = np.argmax(final_preds, axis=1)  

    # Calculate accuracy
    print(f"Accuracy on Test dataset by the combined model: {accuracy_score(Yus_labels, final_preds) * 100:.2f}%")  

    # Confusion matrix
    cf_matrix = confusion_matrix(Yus_labels, final_preds)  
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)  
    plt.title("Confusion Matrix for Combined Model on Test Dataset")  
    plt.show()   

    
    
def evaluate_model(model, Xus, Yus):
    """
    Evaluates the trained model on unseen data.

    Parameters:
    - model: Trained model to evaluate.
    - Xus: DataFrame, features for unseen (test) data.
    - Yus: Series, labels for unseen (test) data.
    """
    
    predictions = model.predict(Xus)

    if predictions.ndim > 1:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = predictions

    true_classes = np.argmax(Yus, axis=1)

    accuracy = accuracy_score(np.asarray(true_classes), predicted_classes)
    print(f'Accuracy: {accuracy}')
    if 1>accuracy>0.9:
        print("Good model")
    print(classification_report(np.asarray(true_classes), predicted_classes))
    
    
def objective(trial):
    # Define hyperparameters to tune
    hidden_size_1 = trial.suggest_int('hidden_size_1', 128, 256)  
    hidden_size_2 = trial.suggest_int('hidden_size_2', 128, 256)
    hidden_size_3 = trial.suggest_int('hidden_size_3', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1,log=True)
    epochs = trial.suggest_int('epochs', 500, 2000) 
    init = trial.suggest_categorical('init', ['he', 'xavier',None])

    model = CustomNeuralNetwork(inp=132, out=41, h1=hidden_size_1,h2= hidden_size_2, h3=hidden_size_3,init=init)
    with tf.device('/GPU:0'):
        model.train(X_train, Y_train, epochs=epochs, learning_rates=learning_rate)

    y_pred = model.predict(X_train)
    loss = model.compute_loss(y_pred, Y_train)

    return loss  
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



def grid_search(X_train, Y_train, param_grid,Xus,Yus):
    best_loss = float('inf')
    best_params = None

    for hidden_size_1 in param_grid['hidden_size_1']:
        for hidden_size_2 in param_grid['hidden_size_2']:
            for hidden_size_3 in param_grid['hidden_size_3']:
                for learning_rate in param_grid['learning_rate']:
                    for mode in param_grid['init']:
                        model = CustomNeuralNetwork(inp=132, out=41,
                                                            h1 = hidden_size_1,
                                                            h2 = hidden_size_2,
                                                            h3 = hidden_size_3,
                                                            init = mode)
                        print(f'Hidden Size 1: {hidden_size_1}, Hidden Size 2: {hidden_size_2}, Hidden Size 3: {hidden_size_3}, Learning Rate: {learning_rate}, Mode: {mode}')  
                        with tf.device('/GPU:0'):
                            model.train(X_train, Y_train, epochs=1000, learning_rates=learning_rate)

                        val_loss = model.compute_loss(model.predict(Xus), Yus)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_params = (hidden_size_1,hidden_size_2,hidden_size_3, learning_rate)

    print(f'Best Loss: {best_loss}, Best Params: {best_params}')
    return best_params



def tuning(X_train, Y_train, Xus, Yus):
    param_grid = {
                'hidden_size_1': [32, 41,64, 128,132,256],
                'hidden_size_2': [32, 41,64, 128,132,256],
                'hidden_size_3': [32, 41,64, 128,132,256],
                'learning_rate': [0.001,0.003,0.006, 0.01,0.03,0.06, 0.1],
                'init': ['he', 'xavier',None]
            }
    
    best_params = grid_search(X_train, Y_train, param_grid,Xus,Yus)

    # Create Optuna study, consider running on gpu for efficiency
    with tf.device('/GPU:0'):
        study = optuna.create_study(direction='minimize')  
        study.optimize(objective, n_trials=100) 
    
    print("Best optuna hyperparameters: ", study.best_params)   

    print("Gridsearch best params :",best_params)



if __name__ == "__main__":
    filepath = os.getcwd() + "\\ressources\data.csv"
    
    df = preprocess(read(filepath))      
    
    X_train, Y_train, Xus, Yus = categorize(df)

    # hyper parameter tuning
    tuning(X_train, Y_train, Xus, Yus)

    model = CustomNeuralNetwork(h1 = 192,h2=192,h3=64,init='xavier')
    model.train(X_train, Y_train, epochs=2500, learning_rates=0.07)
    
    evaluate_model(model, Xus, Yus) 
    prediction(model, [Xus, Yus])
    
   
    
 
