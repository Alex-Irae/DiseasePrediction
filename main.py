import sys
from PyQt5.QtWidgets import (QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QPushButton, QLineEdit, QComboBox, QScrollArea, 
                             QMessageBox, QListWidget, QApplication)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from Result_window import ResultWindow
import threading

from disease_predict import (setX, predict, load_diseases, load_models, load_symptoms)

# Path to the application icon
icon_path = "icon.png"

# List of cardiovascular symptoms
cardiovascular_symptoms = ["fast_heart_rate", "palpitations", "prominent_veins_on_calf",
                           "swollen_blood_vessels", "swollen_legs"]

# List of liver symptoms
liver_symptoms = ["yellowish_skin", "dark_urine", "acute_liver_failure", "fluid_overload",
                  "swelling_of_stomach", "fluid_overload.1"]

# List of reproductive symptoms
reproductive_symptoms = ["abnormal_menstruation", "extra_marital_contacts", "irregular_sugar_level",
                         "receiving_blood_transfusion", "receiving_unsterile_injections",
                         "history_of_alcohol_consumption", "puffy_face_and_eyes", "swollen_extremeties"]

# List of urinary symptoms
urinary_symptoms = ["burning_micturition", "yellow_urine", "yellowing_of_eyes", 
                    "foul_smell_of_urine", "continuous_feel_of_urine", "bladder_discomfort", "polyuria"]

# List of neurological symptoms
neurological_symptoms = ["dizziness", "blurred_and_distorted_vision", "slurred_speech", 
                         "spinning_movements", "loss_of_balance", "unsteadiness", "altered_sensorium",
                         "lack_of_concentration", "visual_disturbances", "coma", "loss_of_smell", "pain_behind_the_eyes"]

# List of musculoskeletal symptoms
musculoskeletal_symptoms = ["swollen_legs", "joint_pain", "muscle_wasting", "back_pain", "neck_pain",
                            "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck",
                            "swelling_joints", "movement_stiffness", "painful_walking", "muscle_pain",
                            "cramps", "weakness_of_one_body_side"]

# List of digestive symptoms
digestive_symptoms = ["pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
                      "irritation_in_anus", "excessive_hunger", "increased_appetite", "stomach_pain",
                      "acidity", "ulcers_on_tongue", "vomiting", "spotting_urination", "loss_of_appetite",
                      "indigestion", "nausea", "diarrhoea", "constipation", "abdominal_pain", 
                      "belly_pain", "stomach_bleeding", "distention_of_abdomen", "passage_of_gases"]

# List of respiratory symptoms
respiratory_symptoms = ["continuous_sneezing", "cough", "breathlessness", "phlegm", "throat_irritation", 
                        "sinus_pressure", "runny_nose", "congestion", "chest_pain", "fast_heart_rate",
                        "palpitations", "mucoid_sputum", "rusty_sputum", "blood_in_sputum"]

# List of skin symptoms
skin_symptoms = ["bruising", "brittle_nails", "pus_filled_pimples", "blackheads", "redness_of_eyes",
                 "patches_in_throat", "swelled_lymph_nodes", "itching", "skin_rash", "nodal_skin_eruptions",
                 "red_spots_over_body", "scurring", "blister", "red_sore_around_nose", "yellow_crust_ooze",
                 "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", 
                 "drying_and_tingling_lips", "dischromic_patches", "internal_itching"]

# List of general symptoms
general_symptoms = ["shivering", "chills", "obesity", "watering_from_eyes", "enlarged_thyroid", "fatigue",
                    "weight_gain", "weight_loss", "high_fever", "mild_fever", "lethargy", "malaise", 
                    "sweating", "dehydration", "restlessness", "mood_swings", "anxiety", "depression",
                    "irritability", "cold_hands_and_feets", "sunken_eyes", "weakness_in_limbs", 
                    "weakness_of_one_body_side", "toxic_look_(typhos)", "family_history", "headache"]

# Load the full list of symptoms from an external source
symptoms_full_list = load_symptoms()

class PredictDis(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set the window title and icon
        self.setWindowTitle("Disease Prediction")
        self.setWindowIcon(QIcon(icon_path))
        
        # Set the geometry of the main window
        self.setGeometry(1000, 500, 1000, 1000)
        
        # Create a QLineEdit for symptom input
        self.symptom_input = QLineEdit(self)
        self.symptom_input.setPlaceholderText("Enter your symptom")
        
        # Create a QPushButton for searching symptoms
        self.search_symp_button = QPushButton("Search", self)
        
        # Create QLabel for menu title and symptom prompt
        self.menu_title = QLabel("Select a category for your symptoms", self)
        self.symptom_prompt = QLabel("Select the symptoms", self)
        
        # Create QPushButton for submitting and resetting symptoms
        self.submit_button = QPushButton("Submit", self)
        self.reset_button = QPushButton("Empty the symptom list")
        
        # Create a QWidget to hold the symptoms panel
        self.symptoms_panel = QWidget(self)
        
        # Create a QScrollArea to make the symptoms panel scrollable
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidget(self.symptoms_panel)
        self.scroll_area.setWidgetResizable(True)
     
        # Set the geometry of the symptoms panel
        self.symptoms_panel.setGeometry(0, int(2*self.height() / 3), self.width(), int(2*self.height() / 3))
       
        # Create a QGridLayout for the symptoms panel
        self.gl = QGridLayout(self.symptoms_panel)
        self.gl.setContentsMargins(0, 0, 0, 0)  
        self.gl.setSpacing(10)  
        
        # Dictionary to store selected symptoms
        self.selected_symptoms = {}
        
        # Create a QListWidget to display selected symptoms
        self.selected_symptoms_list = QListWidget(self)
        
        # Dictionary to store symptom buttons
        self.symptom_buttons = {}
        
        # Initialize the menu and UI in separate threads
        thread1 = threading.Thread(target=self.initmenu())
        thread2 = threading.Thread(target=self.initUI())
        thread1.start()
        thread2.start()
        
        # Set the style of the UI components
        self.setStyle()
        
        # Connect buttons to their respective functions
        self.search_symp_button.clicked.connect(self.search_symptoms)
        self.submit_button.clicked.connect(self.initmodels)
        self.reset_button.clicked.connect(self.reset)
        
    def initUI(self):
        # Initialize the main UI components
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        hbox_input = QHBoxLayout()
        hbox_input.addWidget(self.symptom_input)
        hbox_input.addWidget(self.search_symp_button)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_input)

        self.menu_title.setFixedHeight(30)  
        self.symptom_prompt.setFixedHeight(30)  

        vbox.addWidget(self.menu_title)
        vbox.addWidget(self.menu_categories)
        vbox.addWidget(self.symptom_prompt)
        vbox.addWidget(self.scroll_area)

        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(self.reset_button)
        hbox_buttons.addWidget(self.submit_button)

        hbox_dict = QHBoxLayout()
        hbox_dict.addLayout(vbox)

        self.selected_symptoms_list.setFixedWidth(250)  
        hbox_dict.addWidget(self.selected_symptoms_list)

        vbox_fin = QVBoxLayout()
        vbox_fin.addLayout(hbox_dict)

        vbox_fin.setContentsMargins(10, 10, 10, 10) 
        vbox_fin.setSpacing(5)  

        vbox.addLayout(hbox_buttons)

        central_widget.setLayout(vbox_fin)
        central_widget.setMinimumWidth(1560)
        central_widget.setMaximumHeight(1600)        

    def setStyle(self):
        # Set the style for various UI components
        self.menu_title.setAlignment(Qt.AlignCenter)
        self.symptom_prompt.setAlignment(Qt.AlignCenter)
        self.search_symp_button.setObjectName("search_symp_button")
        self.submit_button.setObjectName("submit_button")
        self.reset_button.setObjectName("reset_button")
        
        self.setStyleSheet("""
            QPushButton#search_symp_button,
            QPushButton#submit_button,
            QPushButton#reset_button {
                font-size: 20px;
                font-family: Century Gothic;
                padding: 25px 15px;  
                margin: 20px; 
                border: 3px solid;
                border-radius: 10px;
            }
            QPushButton#search_symp_button:hover,
            QPushButton#submit_button:hover,
            QPushButton#reset_button:hover {
                background-color: red;
                
            }
           
            QComboBox, QLabel, QLineEdit {
                font-family: Century Gothic;
                margin: 10px;
                padding: 15px 10px; 
            }
        """)

    def initmenu(self):
        # Initialize the symptom category menu
        categories = ["Generic Symptoms", "All symptoms", "Cardiovascular Symptoms", "Liver Symptoms", 
                      "Reproductive Symptoms", "Urinary Symptoms", "Neurological Symptoms",
                      "Musculoskeletal Symptoms", "Digestive Symptoms", 
                      "Respiratory Symptoms", "Skin Symptoms"]
        self.menu_categories = QComboBox(self)
        for category in categories: 
            self.menu_categories.addItem(category)
        self.menu_categories.currentIndexChanged.connect(self.update_symptoms)
        self.update_symptoms()
        
    def update_symptoms(self):
        # Update the symptoms displayed based on the selected category
        selected_index = self.menu_categories.currentIndex()

        while self.gl.count():
            btn = self.gl.takeAt(0).widget()
            if btn:
                btn.deleteLater()

        symptoms_to_display = []
        match selected_index:
            case 0: symptoms_to_display = general_symptoms
            case 1: symptoms_to_display = symptoms_full_list
            case 2: symptoms_to_display = cardiovascular_symptoms
            case 3: symptoms_to_display = liver_symptoms
            case 4: symptoms_to_display = reproductive_symptoms
            case 5: symptoms_to_display = urinary_symptoms
            case 6: symptoms_to_display = neurological_symptoms
            case 7: symptoms_to_display = musculoskeletal_symptoms
            case 8: symptoms_to_display = digestive_symptoms
            case 9: symptoms_to_display = respiratory_symptoms
            case 10: symptoms_to_display = skin_symptoms
            
        self.create_buttons(symptoms_to_display, selected_index)

    def create_buttons(self, symptoms_to_display, selected_index):
        # Create buttons for each symptom in the selected category
        panel_width = self.symptoms_panel.width()  
        if panel_width > (30/35) * self.width():
            panel_width = int((30/35) * self.width())
        button_width = 300  
        columns = panel_width // button_width  

        if columns < 1:
            columns = 1

        self.symptom_buttons.clear()

        for i, symptom in enumerate(symptoms_to_display):
            if "_" in symptom:
                symptom = symptom.replace("_", " ")
            symptom = symptom.capitalize()
            button = QPushButton(symptom)
            button.setStyleSheet("font-size:18px; font-family: Century Gothic; padding: 10px 10px; border: 3px solid; border-radius: 2px;")
            if selected_index == 1:
                button.setMaximumSize(300, 70)
            else: 
                button.setMinimumSize(300, 100)
            button.clicked.connect(self.toggle_button)
            self.gl.addWidget(button, i // columns, i % columns)
            self.symptom_buttons[symptom] = button

            # Set initial button color based on selected symptoms
            if symptom in self.selected_symptoms and self.selected_symptoms[symptom]:
                button.setStyleSheet(button.styleSheet() + " background-color: red;")            

    def search_symptoms(self):
        # Search for a symptom entered in the input field
        sympt = self.symptom_input.text().lower().strip()
        if " " in sympt:
            sympt = sympt.replace(" ", "_")
        if sympt == "":
            self.show_popup(3)
            return
        if sympt in symptoms_full_list:
            if sympt in self.selected_symptoms:
                self.selected_symptoms[sympt] = not self.selected_symptoms[sympt]
            else:
                self.selected_symptoms[sympt] = True

            sympt2 = self.symptom_input.text().capitalize().strip()
            if str(sympt2) in self.symptom_buttons:
                button = self.symptom_buttons[sympt2]
                if self.selected_symptoms[sympt]:
                    button.setStyleSheet(button.styleSheet() + " background-color: red;")
                else:
                    button.setStyleSheet(button.styleSheet().replace("background-color: red;", ""))

            self.update_selected_symptoms_display()
        else:
            self.show_popup(1)

    def toggle_button(self):
        # Toggle the selection state of a symptom button
        symptom = self.sender().text().lower().strip()
        button = self.sender()
        if " " in symptom:
            symptom = symptom.replace(" ", "_")
        if symptom in self.selected_symptoms:
            self.selected_symptoms[symptom] = not self.selected_symptoms[symptom]
        else:
            self.selected_symptoms[symptom] = True

        # Update button style based on current state
        if self.selected_symptoms[symptom]:
            button.setStyleSheet(button.styleSheet() + " background-color: red;")
        else:
            button.setStyleSheet(button.styleSheet().replace("background-color: red;", ""))

        self.update_selected_symptoms_display()

    def update_selected_symptoms_display(self):
        # Update the display of selected symptoms
        self.selected_symptoms_list.clear()
        for symptom, selected in self.selected_symptoms.items():
            if selected:
                symptom = symptom.capitalize()
                if "_" in symptom:
                    symptom = symptom.replace("_", " ")
                self.selected_symptoms_list.addItem(symptom)
                
    def show_popup(self, num):
        # Show a popup message based on the provided number
        match num:
            case 0: message = "Please select at least 1 symptom, the more, the more accurate the prediction will be."
            case 1: message = "This symptom does not exist."
            case 2: message = "You already selected this symptom."
            case 3: message = "Please enter a symptom to search for."

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Warning!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def initmodels(self):
        # Initialize the models and make predictions based on selected symptoms
        symptoms = [symptom for symptom, selected in self.selected_symptoms.items() if selected]
        if not symptoms:
            self.show_popup(0)
            return
        
        disease = load_diseases()
        input_data = setX(symptoms)
        
        models, model_names = load_models()
        predictions, probabilities = predict(input_data, models)
        
        result_window = ResultWindow(predictions, probabilities, model_names, disease, input_data)
        self.reset()
        result_window.exec_()

    def reset(self):
        # Reset the selected symptoms and update the display
        self.selected_symptoms = {}
        self.selected_symptoms_list.clear()
        self.update_selected_symptoms_display()
        self.update_symptoms()
        
if __name__ == "__main__":
    # Run the application
    app = QApplication(sys.argv)
    win = PredictDis()
    win.show()
    sys.exit(app.exec_())