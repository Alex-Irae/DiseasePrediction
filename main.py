import sys
from PyQt5.QtWidgets import (QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QPushButton, QLineEdit, QComboBox, QScrollArea, 
                             QMessageBox, QListWidget, QApplication)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from Result_window import ResultWindow
from disease_predict import load_symptoms

# Path to the application icon
icon_path = "ressources\icon.png"

# Symptom categories dictionary, mapping categories to a list of symptoms
symptom_categories = {
    "Generic Symptoms": ["shivering", "chills", "obesity", "watering_from_eyes", "enlarged_thyroid", "fatigue",
                         "weight_gain", "weight_loss", "high_fever", "mild_fever", "lethargy", "malaise", 
                         "sweating", "dehydration", "restlessness", "mood_swings", "anxiety", "depression",
                         "irritability", "cold_hands_and_feets", "sunken_eyes", "weakness_in_limbs", 
                         "weakness_of_one_body_side", "toxic_look_(typhos)", "family_history", "headache"],
    "All Symptoms": load_symptoms(),
    
    "Cardiovascular Symptoms": ["fast_heart_rate", "palpitations", "prominent_veins_on_calf",
                                "swollen_blood_vessels", "swollen_legs"],
    "Liver Symptoms": ["yellowish_skin", "dark_urine", "acute_liver_failure", "fluid_overload",
                       "swelling_of_stomach", "fluid_overload.1"],
    "Reproductive Symptoms": ["abnormal_menstruation", "extra_marital_contacts", "irregular_sugar_level",
                              "receiving_blood_transfusion", "receiving_unsterile_injections",
                              "history_of_alcohol_consumption", "puffy_face_and_eyes", "swollen_extremeties"],
    "Urinary Symptoms": ["burning_micturition", "yellow_urine", "yellowing_of_eyes", 
                         "foul_smell_of_urine", "continuous_feel_of_urine", "bladder_discomfort", "polyuria"],
    "Neurological Symptoms": ["dizziness", "blurred_and_distorted_vision", "slurred_speech", 
                              "spinning_movements", "loss_of_balance", "unsteadiness", "altered_sensorium",
                              "lack_of_concentration", "visual_disturbances", "coma", "loss_of_smell", "pain_behind_the_eyes"],
    "Musculoskeletal Symptoms": ["swollen_legs", "joint_pain", "muscle_wasting", "back_pain", "neck_pain",
                                 "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck",
                                 "swelling_joints", "movement_stiffness", "painful_walking", "muscle_pain",
                                 "cramps", "weakness_of_one_body_side"],
    "Digestive Symptoms": ["pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
                           "irritation_in_anus", "excessive_hunger", "increased_appetite", "stomach_pain",
                           "acidity", "ulcers_on_tongue", "vomiting", "spotting_urination", "loss_of_appetite",
                           "indigestion", "nausea", "diarrhoea", "constipation", "abdominal_pain", 
                           "belly_pain", "stomach_bleeding", "distention_of_abdomen", "passage_of_gases"],
    "Respiratory Symptoms": ["continuous_sneezing", "cough", "breathlessness", "phlegm", "throat_irritation", 
                             "sinus_pressure", "runny_nose", "congestion", "chest_pain", "fast_heart_rate",
                             "palpitations", "mucoid_sputum", "rusty_sputum", "blood_in_sputum"],
    "Skin Symptoms": ["bruising", "brittle_nails", "pus_filled_pimples", "blackheads", "redness_of_eyes",
                      "patches_in_throat", "swelled_lymph_nodes", "itching", "skin_rash", "nodal_skin_eruptions",
                      "red_spots_over_body", "scurring", "blister", "red_sore_around_nose", "yellow_crust_ooze",
                      "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", 
                      "drying_and_tingling_lips", "dischromic_patches", "internal_itching"]
    
}

# Load the full list of symptoms from an external source
symptoms_full_list = load_symptoms()

class PredictDis(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize the main window properties
        self.setWindowTitle("Disease Prediction")
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(600, 175, 1400, 1300)
        
        # UI components
        self.symptom_input = QLineEdit(self)
        self.symptom_input.setPlaceholderText("Enter your symptom")
        self.search_symp_button = QPushButton("Search", self)
        self.menu_title = QLabel("Select a category for your symptoms", self)
        self.symptom_prompt = QLabel("Select the symptoms", self)
        self.submit_button = QPushButton("Submit", self)
        self.reset_button = QPushButton("Empty the symptom list")
        
        self.symptoms_panel = QWidget(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidget(self.symptoms_panel)
        self.scroll_area.setWidgetResizable(True)
        self.symptoms_panel.setGeometry(0, int(2 * self.height() / 3), self.width(), int(2 * self.height() / 3))
        
        self.gl = QGridLayout(self.symptoms_panel)
        self.gl.setContentsMargins(0, 0, 0, 0)
        self.gl.setSpacing(10)
        
        self.menu_categories = QComboBox(self)
        self.selected_symptoms = {}
        self.selected_symptoms_list = QListWidget(self)
        self.symptom_buttons = {}
        
        # Initialize the UI, menu, and styles
        self.initUI()
        self.initMenu()
        self.setStyle()
        
        # Connect buttons to functions
        self.search_symp_button.clicked.connect(self.search_symptoms)
        self.submit_button.clicked.connect(self.initModels)
        self.reset_button.clicked.connect(self.reset)
        
    def initUI(self):
        """
        Initialize the main layout of the window.
        Input: None
        Output: None (sets up the UI components)
        """
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
        """
        Set the style for various UI components.
        Input: None
        Output: None (applies stylesheet to components)
        """
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

    def initMenu(self):
        """
        Initialize the symptom category menu.
        Input: None
        Output: None (populates the symptom categories)
        """
        for category in symptom_categories.keys():
            self.menu_categories.addItem(category)
        self.menu_categories.currentIndexChanged.connect(self.updateSymptoms)
        self.updateSymptoms()

    def updateSymptoms(self):
        """
        Update the symptoms displayed based on the selected category.
        Input: None
        Output: None (updates symptom buttons in UI)
        """
        selected_index = self.menu_categories.currentIndex()
        category = list(symptom_categories.keys())[selected_index]
        symptoms_to_display = symptom_categories[category]

        while self.gl.count():
            btn = self.gl.takeAt(0).widget()
            if btn:
                btn.deleteLater()

        self.createButtons(symptoms_to_display)

    def createButtons(self, symptoms_to_display):
        """
        Create buttons for each symptom in the selected category.
        Input: symptoms_to_display (list of symptoms in the category)
        Output: None (creates symptom buttons and adds them to UI)
        """
        panel_width = self.symptoms_panel.width()
        if panel_width > (30 / 35) * self.width():
            panel_width = int((30 / 35) * self.width())
        button_width = 300
        columns = panel_width // button_width

        if columns < 1:
            columns = 1

        self.symptom_buttons.clear()

        for i, symptom in enumerate(symptoms_to_display):
            s = symptom.strip()
            symptom = symptom.replace("_", " ").capitalize()
            button = QPushButton(symptom)
            button.setStyleSheet("font-size:18px; font-family: Century Gothic; padding: 10px 10px; border: 3px solid; border-radius: 2px;")
            button.setMinimumSize(300, 100)
            button.clicked.connect(self.toggleButton)
            self.gl.addWidget(button, i // columns, i % columns)
            self.symptom_buttons[symptom] = button

            if s in self.selected_symptoms:
                if self.selected_symptoms[s]:
                    button.setStyleSheet(button.styleSheet() + " background-color: red;")

    def search_symptoms(self):
        """
        Search for a symptom entered by the user.
        Input: None (takes the user-entered symptom from QLineEdit)
        Output: None (adds symptom if found, shows popup otherwise)
        """
        sympt = self.symptom_input.text().lower().strip().replace(" ", "_")
        if not sympt:
            self.showPopup(3)
            return
        if sympt in symptoms_full_list:
            self.selected_symptoms[sympt] = not self.selected_symptoms.get(sympt, False)
            self.updateButtonStyle(sympt)
            self.updateSelectedSymptomsDisplay()
        else:
            self.showPopup(1)

    def toggleButton(self):
        """
        Toggle the selection state of a symptom button.
        Input: None (connected to button click event)
        Output: None (toggles selected state and updates UI)
        """
        symptom = self.sender().text().lower().strip().replace(" ", "_")
        self.selected_symptoms[symptom] = not self.selected_symptoms.get(symptom, False)
        self.updateButtonStyle(symptom)
        self.updateSelectedSymptomsDisplay()

    def updateButtonStyle(self, symptom):
        """
        Update the style of a symptom button based on its selection state.
        Input: symptom (str) - symptom name to update
        Output: None (changes the button color based on selection)
        """
        capitalized_symptom = symptom.replace("_", " ").capitalize()
        button = self.symptom_buttons[capitalized_symptom]
        if self.selected_symptoms[symptom]:
            button.setStyleSheet(button.styleSheet() + " background-color: red;")
        else:
            button.setStyleSheet(button.styleSheet().replace(" background-color: red;", ""))

    def updateSelectedSymptomsDisplay(self):
        """
        Update the display of selected symptoms in the UI.
        Input: None
        Output: None (updates QListWidget with selected symptoms)
        """
        self.selected_symptoms_list.clear()
        for symptom, selected in self.selected_symptoms.items():
            if selected:
                self.selected_symptoms_list.addItem(symptom.replace("_", " ").capitalize())

    def initModels(self):
        """
        Initialize machine learning models with selected symptoms.
        Input: None
        Output: None (processes selected symptoms and predicts diseases)
        """
        if not any(self.selected_symptoms.values()):
            self.showPopup(2)
        else:
            selected_symptoms_list = [k for k, v in self.selected_symptoms.items() if v]
            self.predict(selected_symptoms_list)

    def reset(self):
        """
        Reset the selected symptoms and UI.
        Input: None
        Output: None (clears selection and resets button states)
        """
        self.selected_symptoms = {k: False for k in self.selected_symptoms}
        self.updateSelectedSymptomsDisplay()
        self.updateSymptoms()
        

    def showPopup(self, error_code):
        """
        Display an error message popup based on error_code.
        Input: error_code (int) - determines the type of message to show
        Output: None (displays a QMessageBox with relevant information)
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        if error_code == 1:
            msg.setText("Symptom not found.")
        elif error_code == 2:
            msg.setText("Please select at least one symptom.")
        elif error_code == 3:
            msg.setText("Symptom input cannot be empty.")
        msg.setWindowTitle("Error")
        msg.exec_()

    def predict(self, selected_symptoms):
        """
        Perform disease prediction based on selected symptoms.
        Input: selected_symptoms (list of str) - list of symptoms for prediction
        Output: None (displays result in a new window)
        """
        self.result_window = ResultWindow(selected_symptoms)
        self.result_window.show()
        self.reset()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictDis()
    window.show()
    sys.exit(app.exec_())
