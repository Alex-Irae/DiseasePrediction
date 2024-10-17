import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout,QDialog,QPushButton,QMessageBox
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import subprocess


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanva


class ResultWindow(QDialog):
    def __init__(self, predictions, probabilities, models, disease_dict,X):
        super().__init__()
        self.X = X
        self.setWindowTitle("Prediction Results")
        self.setGeometry(1000,500,1000,1000)
        layout = QVBoxLayout()

        self.add_button = QPushButton("Add Prediction", self)
        
        self.disease_guess = []
        self.disease_probs = []
        for idx, model_name in enumerate(models):
            disease_probs = probabilities[idx]
            prediction = predictions[idx]
            disease_name = disease_dict[str(prediction)]
            self.disease_guess.append(disease_name)
            self.disease_probs.append(disease_probs.argmax())
            section_widget = self.create_result_section(model_name, disease_probs, disease_dict, disease_name)
            layout.addWidget(section_widget)
        layout.addStretch()
        layout.addWidget(self.add_button,alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.setLayout(layout)
        
        self.add_button.clicked.connect(lambda: self.pop_add("Do you want to add your data to be trained? This action cannot be undone.",1))

    def create_result_section(self, model_name, disease_probs, disease_dict, predicted_disease):

        section_widget = QWidget()
        section_layout = QVBoxLayout(section_widget)

        title_label = QLabel(f"Model: {model_name}")
        title_label.setStyleSheet("""font-size: 20px; 
                                  font-weight: bold;
                                  font-family: Century Gothic;""")
        section_layout.addWidget(title_label)
        self.add_button.setStyleSheet("""font-size: 20px;
                                        font-family: Century Gothic;
                                        padding: 25px 15px;  
                                        margin: 20px; 
                                        border: 3px solid;
                                        border-radius: 10px;""")
        
        
        figure, ax = plt.subplots(figsize=(5, 3))
        canvas = FigureCanva(figure)

        filtered_probs = {disease_dict[str(i)]: prob for i, prob in enumerate(disease_probs) if prob > 0.1}
        if not filtered_probs:
            max_index = disease_probs.argmax()
            filtered_probs[disease_dict[str(max_index)]] = max(disease_probs)

        diseases = list(filtered_probs.keys())
        probs = list(filtered_probs.values())

        ax.barh(diseases, probs, color='skyblue')
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)

        section_layout.addWidget(canvas)

        pred_label = QLabel(f"Predicted disease: {predicted_disease}\n Probability: {max(probs)*100:.2f} %")
        pred_label.setStyleSheet("font-size: 20px;font-family: Century Gothic;")
        section_layout.addWidget(pred_label)

        return section_widget
    def pop_add(self,message,ind = 0):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Warning!")
        if ind!= 0 :
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No )
            result = msg.exec_()
            if result == QMessageBox.Yes:
                self.add_data()
            else:
                return
        else :
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
                
    def add_data(self):
        for disease in self.disease_guess:
            
            if self.disease_guess.count(disease) == 1:
                self.disease_guess.remove(disease)
                self.disease_probs.remove(self.disease_probs[self.disease_guess.index(disease)])
                
            if self.disease_guess.count(disease) ==2 :
                print(self.disease_probs)
                for prob in self.disease_probs:
                    if prob > 0.5:
                        self.pop_add("The statistics are too uncertain to train, please try again later.")
                        return
                    else :
                        pass
        if len(self.disease_guess) == 0:
                self.pop_add("The statistics are too uncertain to train, please try again later.")
                return
        else :
            disease_name = self.disease_guess[0]
            X = self.X
            data_to_add = pd.DataFrame([list(X) + [disease_name]])


            data_to_add.to_csv('data.csv', mode='a', header=False, index=False)
            self.pop_add("The data have been added to the training data. Please wait for the training process to complete.")
            self.run_predict_trainer()
        
        
    def run_predict_trainer(self):
        # subprocess.run(['python', 'PredictTrainer.py'])
        self.pop_add("Training complete, Please reload the application to see the updated results.  ")
