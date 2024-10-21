import pandas as pd
import json
import numpy as np
from PyQt5.QtWidgets import (QLabel, QWidget, QVBoxLayout, QDialog, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
import PyQt5
import subprocess
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanva
import matplotlib.pyplot as plt
import seaborn as sns
from disease_predict import (setX, predict, load_diseases, load_models)

class ResultWindow(QDialog):
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
        self.models, model_names = load_models()

        # Ensure the voting classifier is the last in the model list
        if "voting_classifier" in model_names:
            idx = model_names.index("voting_classifier")
            model_names.append(model_names.pop(idx))
            self.models.append(self.models.pop(idx))

        # Prepare feature vector for prediction
        self.X = setX(symptoms)
        predictions, probabilities = predict(self.X, self.models)

        # Configure the window title and layout
        self.setWindowTitle("Prediction Results")
        self.setGeometry(600, 250, 1200, 1200)
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
        for idx, model_name in enumerate(model_names):
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
        self.recom_button.clicked.connect(lambda: self.display_recommendation(self.disease))

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

    def display_recommendation(self, disease):
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
        
        # Show Yes/No if `ind` is set, else just Ok
        if ind != 0:  # To add data
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
            if self.disease_probs[self.disease_guess.index(disease)] > 0.95:
                subprocess.run(["python", "add_to_database.py", disease] + self.symptoms, shell=True)
                self.pop_up("The prediction has been added")
                break

    def plot_feature_importance(self, dialog, feature_names):
        """
        Plots the feature importance (weights) for each symptom in a bar chart.
        
        Args:
            dialog (QDialog): The dialog window in which the plot is displayed.
            feature_names (list): List of symptom feature names.
        """
        layout = QVBoxLayout(dialog)
        figure, ax = plt.subplots(figsize=(10, 8))
        canvas = FigureCanva(figure)
        sns.barplot(x=feature_names, y=self.models[-1].estimators_[0].coef_[0], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        plt.close()
