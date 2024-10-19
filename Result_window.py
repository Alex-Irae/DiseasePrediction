import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
import subprocess
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanva
import matplotlib.pyplot as plt
import seaborn as sns
from disease_predict import (setX, predict, load_diseases, load_models)



class ResultWindow(QDialog):
    def __init__(self, symptoms):
        super().__init__()

        self.symptoms = symptoms
        
        disease_dict = load_diseases()
        self.models, model_names = load_models()
        
        # to only display the classifier uncomment those
        x = model_names.index('voting_classifier')
        self.models = [self.models[x]]
        model_names = [model_names[x]]
        
        self.X = setX(symptoms)
        predictions, probabilities = predict(self.X, self.models)
        
        self.setWindowTitle("Prediction Results")
        self.setGeometry(1000, 500, 1000, 1000)
        layout = QVBoxLayout()
        
        # Button to add prediction to the training data
        self.add_button = QPushButton("Add Prediction", self)
        
        self.weight_button = QPushButton("Symptoms influence", self)
        if len(self.models) == 1:
            self.weight_button.hide()
        
        self.disease_guess = []
        self.disease_probs = []
        
        # Iterate over each model to display its prediction and probabilities
        for idx, model_name in enumerate(model_names):
            disease_probs = probabilities[idx]
            prediction = predictions[idx]
            disease_name = disease_dict[str(prediction)]
            self.disease_guess.append(disease_name)
            self.disease_probs.append(disease_probs.argmax())
            section_widget = self.create_result_section(model_name, disease_probs, disease_dict, disease_name)
            layout.addWidget(section_widget)
        
        layout.addStretch()
        vbox = QVBoxLayout()
        
        vbox.addWidget(self.weight_button)
        vbox.addWidget(self.add_button)
        layout.addLayout(vbox)

        self.setLayout(layout)
        
        # Connect the add button to the pop_up method
        self.add_button.clicked.connect(lambda: self.pop_up("Do you want to add your data to be trained? This action cannot be undone.", 1))
        self.weight_button.clicked.connect(self.show_weight)

    def create_result_section(self, model_name, disease_probs, disease_dict, predicted_disease):
        # Create a section to display the results of a model
        section_widget = QWidget()
        section_layout = QVBoxLayout(section_widget)

        # Title label for the model
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
                                        border-radius: 10px;
                                        """)
        self.weight_button.setStyleSheet("""font-size: 20px;
                                        font-family: Century Gothic;
                                        padding: 25px 15px;  
                                        margin: 20px; 
                                        border: 3px solid;
                                        border-radius: 10px;
                                        """)
        
        # Create a bar chart to display the probabilities
        figure, ax = plt.subplots(figsize=(5, 3))
        canvas = FigureCanva(figure)

        # Filter probabilities to only show those greater than 0.1
        filtered_probs = {disease_dict[str(i)]: prob for i, prob in enumerate(disease_probs) if prob >0.15}
        if not filtered_probs:
            max_index = disease_probs.argmax()
            filtered_probs[disease_dict[str(max_index)]] = max(disease_probs)

        diseases = list(filtered_probs.keys())
        probs = list(filtered_probs.values())

        ax.barh(diseases, probs, color='skyblue')
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)

        section_layout.addWidget(canvas)

        # Label to display the predicted disease and its probability
        pred_label = QLabel(f"Predicted disease: {predicted_disease}\n Probability: {max(probs) * 100:.2f} %")
        pred_label.setStyleSheet("font-size: 20px;font-family: Century Gothic;")
        section_layout.addWidget(pred_label)
        plt.close()
        return section_widget



    def pop_up(self, message, ind=0):
        # Show a popup message with an optional Yes/No prompt
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Warning!")
        if ind != 0: # to add data
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
        string_index = 0
        feature_names = self.X.copy()
        
        for i in range(len(feature_names)):
            if feature_names[i] == 1:
                feature_names[i] = self.symptoms[string_index]
                string_index += 1
            

        plot_dialog = QDialog(self)
        plot_dialog.setWindowTitle("Feature Importances")
        plot_dialog.setGeometry(1000, 500, 800, 600)
        
        # Plot feature importance here
        self.plot_feature_importance(plot_dialog, feature_names)

        plot_dialog.show()  # Show as non-modal dialog
        
        
    def add_data(self):
        # Add the prediction data to the training dataset
        for disease in self.disease_guess:
            if self.disease_guess.count(disease) == 1:
                self.disease_guess.remove(disease)
                self.disease_probs.remove(self.disease_probs[self.disease_guess.index(disease)])
                
            if self.disease_guess.count(disease) == 2:
                print(self.disease_probs)
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
            # data_to_add.to_csv('data.csv', mode='a', header=False, index=False)
            self.pop_up("The data have been added to the training data. Please wait for the training process to complete.")
            self.run_predict_trainer()

    def run_predict_trainer(self):
        # Uncomment to run the training script to update the model
        # subprocess.run(['python', 'PredictTrainer.py'])
        self.pop_up("Training complete, Please reload the application to see the updated results.")


    def plot_feature_importance(self, dialog, feature_names):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        
    
        lr_model,rf_model, svm_model, voting_classifier, xgb_model = self.models
        to_plot = [ lr_model,rf_model, xgb_model,]
        model_names = ["Logistic Regression","Random Forest", "XGBoost"]

        for i, model in enumerate(to_plot):
            model_name = model_names[i]
            
            if model_name == "XGBoost" or model_name== "Random Forest":
                
                importances = model.feature_importances_
                importances = [imp for imp, feature in zip(importances, feature_names) if feature != 0]

                feature_names = [feature for feature in feature_names if feature != 0]
                                
                sns.barplot(ax=axes[i], x=feature_names, y=importances, palette="viridis",hue=feature_names,legend=False)
       
       
            elif model_name == "Logistic Regression":                
                coef = model.coef_[0]

                filtered_coef = [c for c, feature in zip(coef, feature_names) if feature != 0]
                filtered_feature_names = [feature for feature in feature_names if feature != 0]

                indices = np.argsort(filtered_coef)[::-1]

                sns.barplot(ax=axes[i],  x=np.array(filtered_feature_names)[indices],  y=np.array(filtered_coef)[indices],  palette="viridis", hue=np.array(filtered_feature_names)[indices],  legend=False)
                
            axes[i].set_title(f'Feature Importances: {model_name}')
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel('Importance')
            
            axes[i].tick_params(axis='x', rotation=45)  
        
        plt.tight_layout()  
        
        canvas = FigureCanva(fig)  
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)


if __name__ == "__main__":
    subprocess.run(['python', 'main.py'])