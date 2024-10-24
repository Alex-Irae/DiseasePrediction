import pandas as pd
import json
import numpy as np
from PyQt5.QtWidgets import (QLabel, QWidget, QVBoxLayout, QDialog, QPushButton, QMessageBox,QApplication)
from PyQt5.QtCore import Qt
import PyQt5
import subprocess
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanva
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (setX, predict, load_diseases, load_models)

class Ml_ResultWindow(QDialog):
    """
    A window that displays disease prediction results based on the provided symptoms.
    
    Attributes:
        symptoms (list): A list of symptoms to predict the disease.
        models (list): List of loaded machine learning models.
        X (ndarray): Feature vector created from symptoms.
        disease_guess (list): List of predicted diseases.
        disease_probs (list): List of predicted probabilities.
        disease (str): The final predicted disease from the voting classifier.
    """

    def __init__(self, symptoms):
        """
        Initializes the result window, sets up the layout, and handles model predictions.
        
        Args:
            symptoms (list): List of symptoms used for disease prediction.
        """
        super().__init__()

        # Store passed symptoms and load disease dictionary and models
        self.symptoms = symptoms
        self.setWindowIcon(PyQt5.QtGui.QIcon("ressources\\icon.png"))

        disease_dict = load_diseases()
        self.models, self.model_names = load_models()

        # Ensure the voting classifier is the last in the model list
        if "voting_classifier" in self.model_names:
            idx = self.model_names.index("voting_classifier")
            self.model_names.append(self.model_names.pop(idx))
            self.models.append(self.models.pop(idx))

        # Prepare feature vector for prediction
        self.X = setX(symptoms)
        predictions, probabilities = predict(self.X, self.models)

        # Configure the window title and layout
        self.setWindowTitle("Prediction Results")
        self.setGeometry(600, 175, 1400, 1300)
        layout = QVBoxLayout()

        # Buttons for additional actions
        self.add_button = QPushButton("Add Prediction", self)
        self.recom_button = QPushButton("Check recommendations", self)
        self.weight_button = QPushButton("Symptoms influence", self)
        if len(self.models) == 1:
            self.weight_button.hide()

        # Label for symptoms
        self.sympt_label = QLabel("Symptoms: " + ', '.join(symptoms.strip().capitalize().replace("_", " ") for symptoms in symptoms))
        self.sympt_label.setAlignment(Qt.AlignCenter)
        self.sympt_label.setStyleSheet("font-weight: bold; font-size: 25px; ")
        layout.addWidget(self.sympt_label)

        # Apply styles to buttons and labels
        self.setStyleSheet("""
            QPushButton{
                font-size: 20px;
                font-family: Century Gothic;
                padding: 25px 15px;  
                margin: 20px; 
                border: 3px solid;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: red;
            } 
            QLabel{
                font-size: 20px;
                font-family: Century Gothic;
                }""")

        # Initialize prediction and probability lists
        self.disease_guess = []
        self.disease_probs = []

        # Display predictions for each model
        for idx, model_name in enumerate(self.model_names):
            disease_probs = probabilities[idx]
            prediction = predictions[idx]
            disease_name = disease_dict[str(prediction)]

            self.disease_guess.append(disease_name)
            self.disease_probs.append(disease_probs.argmax())

            # Create a section to display prediction results for each model
            section_widget = self.create_result_section(model_name, disease_probs, disease_dict, disease_name)
            layout.addWidget(section_widget)

            # If the model is a voting classifier, store the final disease prediction
            if model_name == "voting_classifier":
                self.disease = disease_name
        if not "voting_classifier" in self.model_names:
            self.disease = self.disease_guess[0]
        layout.addStretch()

        # Add buttons for additional actions 
        vbox = QVBoxLayout()
        vbox.addWidget(self.weight_button)
        vbox.addWidget(self.recom_button)
        vbox.addWidget(self.add_button)
        layout.addLayout(vbox)

        self.setLayout(layout)

        # Connect buttons to their respective functions
        self.add_button.clicked.connect(lambda: self.pop_up("Do you want to add your data to be trained? This action cannot be undone.", 1))
        self.weight_button.clicked.connect(self.show_weight)
        self.recom_button.clicked.connect(lambda: self.display_recommendation())

    def create_result_section(self, model_name, disease_probs, disease_dict, predicted_disease):
        """
        Creates a section to display the results for a specific model, including 
        a probability bar chart and predicted disease.
        
        Args:
            model_name (str): The name of the model.
            disease_probs (list): List of probabilities for each disease.
            disease_dict (dict): Dictionary mapping disease indices to names.
            predicted_disease (str): The name of the predicted disease.
        
        Returns:
            QWidget: A widget containing the model's result section.
        """
        section_widget = QWidget()
        section_layout = QVBoxLayout(section_widget)
        
        # Set model title
        model_name = model_name.replace("_model", " ").capitalize()
        title_label = QLabel(f"Model: {model_name.capitalize().replace('_', ' ')}")
        title_label.setStyleSheet("""font-size: 25px; font-weight: bold; font-family: Century Gothic;""")
        section_layout.addWidget(title_label)

        # Create probability bar chart
        figure, ax = plt.subplots(figsize=(5, 3))
        canvas = FigureCanva(figure)

        # Filter diseases with a probability above 0.15
        filtered_probs = {disease_dict[str(i)]: prob for i, prob in enumerate(disease_probs) if prob > 0.15}
        if not filtered_probs:
            max_index = disease_probs.argmax()
            filtered_probs[disease_dict[str(max_index)]] = max(disease_probs)

        diseases = list(filtered_probs.keys())
        probs = list(filtered_probs.values())

        # Plot horizontal bar chart
        ax.barh(diseases, probs, color='red')
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        section_layout.addWidget(canvas)

        # Display predicted disease and its probability
        pred_label = QLabel(f"Predicted disease: {predicted_disease}\n Probability: {max(probs) * 100:.2f} %")
        pred_label.setStyleSheet("font-size: 20px;font-family: Century Gothic;")
        section_layout.addWidget(pred_label)

        plt.close()
        return section_widget

    def display_recommendation(self):
        """
        Displays recommendations based on the predicted disease.
        
        Args:
            disease (str): The name of the predicted disease.
        """
        with open('ressources\\disease_recommendation.json', 'r') as file:
            diseases_recommendation = json.load(file)
        
        # Show a message box with the recommendation for the predicted disease
        msg = QMessageBox()
        msg.setWindowIcon(PyQt5.QtGui.QIcon("ressources\\icon.png"))
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Recommendation!")
        msg.setText(f"Recommendation for {self.disease} \n {diseases_recommendation[self.disease]}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def pop_up(self, message, ind=0):
        """
        Displays a popup message with an optional Yes/No prompt.
        
        Args:
            message (str): The message to display.
            ind (int): Indicator whether to show Yes/No (1) or just Ok (0).
        """
        msg = QMessageBox()
        msg.setWindowIcon(PyQt5.QtGui.QIcon("ressources\\icon.png"))
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Warning!")
        
        if ind != 0:  
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = msg.exec_()
            if result == QMessageBox.Yes:
                self.add_data()
            else:
                return
        else:
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def show_weight(self):
        """
        Displays the influence of symptoms (feature importance) using bar plots.
        """
        string_index = 0
        feature_names = self.X.copy()
        
        # Map symptom names to feature vector
        for i in range(len(feature_names)):
            if feature_names[i] == 1:
                feature_names[i] = self.symptoms[string_index]
                string_index += 1

        # Create dialog window for plotting feature importance
        plot_dialog = QDialog(self)
        plot_dialog.setWindowIcon(PyQt5.QtGui.QIcon("ressources\\icon.png"))
        plot_dialog.setWindowTitle("Feature Importances")
        plot_dialog.setGeometry(750, 400, 1200, 900)

        # Plot feature importance
        self.plot_feature_importance(plot_dialog, feature_names)
        plot_dialog.show()

    def add_data(self):
        """
        Adds the prediction data to the training dataset, checks for confidence.
        """        
        for disease in self.disease_guess:
            if self.disease_guess.count(disease) == 1:
                self.disease_probs.remove(self.disease_probs[self.disease_guess.index(disease)])
                self.disease_guess.remove(disease)
                
            if self.disease_guess.count(disease) == 2:
                for prob in self.disease_probs:
                    if prob > 0.25:
                        self.pop_up("The statistics are too uncertain to train, please try again later.")
                        return
                    else:
                        pass
        if len(self.disease_guess) == 0:
            self.pop_up("The statistics are too uncertain to train, please try again later.")
            return
        else:
            disease_name = self.disease_guess[0]
            X = self.X
            data_to_add = pd.DataFrame([list(X) + [disease_name]])

            data_to_add.to_csv('\\ressources\data.csv', mode='a', header=False, index=False)
            self.pop_up("The data have been added to the training data. Please wait for the training process to complete.")
            self.run_predict_trainer()

    def run_predict_trainer(self):
        """	
        run the training script to update the model
        """	
        subprocess.run(['python', 'MlTrainer.py'])
        subprocess.runn(['python', 'CnnTrainer.py'])
        self.pop_up("Training complete, the application will now restart")
        subprocess.run(['python', 'main.py'])


    def plot_feature_importance(self, dialog, feature_names):
        """
        Plots the feature importance (weights) for each symptom in a bar chart.
        
        Args:
            dialog (QDialog): The dialog window in which the plot is displayed.
            feature_names (list): List of symptom feature names.
        """
        to_plot = []  # List to store the models to plot

        # Collect the models that will be plotted
        for model_name, model in zip(self.model_names, self.models):
            if model_name in ["lr_model", "rf_model", "xgb_model"]:
                to_plot.append((model_name, model))

        # Create subplots based on the number of models in to_plot
        fig, axes = plt.subplots(1, len(to_plot), figsize=(18, 6))  
        fig.suptitle('Feature Importances', fontsize=20)

        # If there's only one model, axes will not be a list
        if len(to_plot) == 1:
            axes = [axes]  # Make it a list for consistency

        for i, (model_name, model) in enumerate(to_plot):
            if model_name == "rf_model":
                importances = model.named_steps['rf'].feature_importances_
            elif model_name == "xgb_model":
                importances = model.named_steps['xgb'].feature_importances_
            elif model_name == "lr_model":
                importances = model.named_steps['lr'].coef_[0]

            indices = np.argsort(importances)[::-1]
            filtered_feature_names = np.array(feature_names)[indices]
            filtered_importances = importances[indices]

            # Filter out the feature named '0'
            mask = filtered_feature_names != '0'
            filtered_feature_names = filtered_feature_names[mask]
            filtered_importances = filtered_importances[mask]

            # Ensure we plot only the available features
            if len(filtered_feature_names) > 0:
                sns.barplot(ax=axes[i], x=filtered_feature_names, y=filtered_importances, palette="viridis",hue = filtered_feature_names)
                axes[i].set_title(f'Feature Importances: {model_name}')
                axes[i].set_xlabel('Features')
                axes[i].set_ylabel('Importance')
                axes[i].tick_params(axis='x', rotation=45)
            else:
                axes[i].set_title(f'No Features for: {model_name}')
                axes[i].set_xlabel('Features')
                axes[i].set_ylabel('Importance')

        plt.tight_layout()

        # Create and display the predicted disease label
        self.disease_label = QLabel(f"Predicted disease: {self.disease}", dialog)
        self.disease_label.setStyleSheet("font-size: 20px; font-family: Century Gothic; font-weight: bold; margin: 10px;")
        self.disease_label.setAlignment(Qt.AlignCenter)
        self.disease_label.setFixedHeight(50)
        
        # Create a canvas for the plot
        canvas = FigureCanva(fig)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.disease_label)
        layout.addWidget(canvas)

