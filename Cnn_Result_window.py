import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanva
import matplotlib.pyplot as plt
from utils import (load_diseases, load_symptoms,load_cnn)
from utils import setX
import json
import PyQt5
import subprocess


class Cnn_ResultWindow(QDialog):
    def __init__(self, symptoms):
        super().__init__()

        self.symptoms = symptoms
        
        disease_dict = load_diseases()
  
        self.cnn_model, self.model_name  = load_cnn(learning_rate=0.07) 
        
        
        self.X = setX(symptoms)
        
        self.prediction =  self.cnn_model.predict(self.X)[0].tolist()

        dis_ind = self.prediction.index((max(self.prediction)))
        
        self.disease_name = disease_dict[str(dis_ind)]
        self.disease_guess = [i for i in range(len(self.prediction)) if self.prediction[i] > 0.2]
        self.prediction = [pred for pred in self.prediction if pred > 0.2]
        self.disease_guess = [disease_dict[str(i)] for i in self.disease_guess]

        self.setWindowTitle("CNN Prediction Results")
        self.setGeometry(0, 500, 1000, 1000)
        self.setWindowIcon(PyQt5.QtGui.QIcon("ressources\\icon.png"))

        layout = QVBoxLayout()
        
        self.sympt_label = QLabel("Symptoms: " + ', '.join(symptoms.strip().capitalize().replace("_", " ") for symptoms in symptoms))
        self.sympt_label.setAlignment(Qt.AlignCenter)
        self.sympt_label.setStyleSheet("font-weight: bold; font-size: 25px; ")
        layout.addWidget(self.sympt_label)
        # Button to add prediction to the training data
        self.add_button = QPushButton("Add Prediction", self)
        self.recom_button = QPushButton("Check recommendations",self)

        
        self.disease_probs = []
        section_widget = self.create_result_section()
        layout.addWidget(section_widget)

                
        layout.addStretch()
        vbox = QVBoxLayout()
        
        vbox.addWidget(self.recom_button)
        vbox.addWidget(self.add_button)
        layout.addLayout(vbox)
        self.setLayout(layout)
        
        # Connect the add button to the pop_up method
        self.add_button.clicked.connect(lambda: self.pop_up("Do you want to add your data to be trained? This action cannot be undone.", 1))
        self.recom_button.clicked.connect(lambda: self.display_recommendation(self.disease_name))

    def create_result_section(self):
        
        section_widget = QWidget()
        section_layout = QVBoxLayout(section_widget)
        
        title_label = QLabel(f"Model: {self.model_name.capitalize().replace('_', ' ')}")
        title_label.setStyleSheet("""font-size: 25px; font-weight: bold; font-family: Century Gothic;""")
        section_layout.addWidget(title_label)

        title_label.setStyleSheet("""font-size: 20px; 
                                  font-weight: bold;
                                  font-family: Century Gothic;""")
        section_layout.addWidget(title_label)
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
            } """)
   
        
        figure, ax = plt.subplots(figsize=(5, 3))
        canvas = FigureCanva(figure)
        

        probs = self.prediction

        probs.append(1 - sum(probs))
        diseases = self.disease_guess
        diseases.append('Other')
        
        ax.pie(probs, labels=diseases, autopct='%1.1f%%', startangle=90, colors=['red', 'blue', 'green', 'yellow', 'purple'])
        ax.axis('equal') 
        section_layout.addWidget(canvas)

        pred_label = QLabel(f"Predicted disease: {self.disease_name}\n Probability: {max(probs) * 100:.2f} %")
        pred_label.setStyleSheet("font-size: 20px;font-family: Century Gothic;")
        section_layout.addWidget(pred_label)

        plt.close()
        return section_widget

    def display_recommendation(self,disease):
        with open('ressources\disease_recommendation.json', 'r') as file:
            diseases_recommendation = json.load(file)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Recommendation!")
        msg.setText(f"Recommendation for {disease} \n {diseases_recommendation[disease]}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


    def pop_up(self, message, ind=0):
        msg = QMessageBox()
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
        
        
        
    def add_data(self):
        # Add the prediction data to the training dataset
        for disease in self.disease_guess:
            if self.disease_guess.count(disease) == 1:
                self.disease_guess.remove(disease)
                self.disease_probs.remove(self.disease_probs[self.disease_guess.index(disease)])
                
            if self.disease_guess.count(disease) == 2:
                for prob in self.disease_probs:
                    if prob > 0.5:
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

            # uncomment to Append the new data to the CSV file
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

   

