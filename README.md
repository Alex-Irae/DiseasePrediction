⚕️Disease Prediction System 🏥

⚕️Overview

This project implements a machine learning-based system to predict diseases based on a list of symptoms. It utilizes various classification algorithms, including SVM, Random Forest, XGBoost, and Logistic Regression, to analyze the input data and provide accurate predictions. The system also incorporates techniques for data balancing, hyperparameter tuning, and model evaluation.


⚕️Features
Data Preprocessing: Reads data from CSV, handles missing values, and encodes categorical labels.
Data Balancing: Utilizes SMOTE to oversample minority classes and create a balanced dataset.
Model Training: Trains multiple models using cross-validation and selects the best performing model.
Hyperparameter Tuning: Optimizes model parameters using GridSearchCV for improved accuracy.
Model Evaluation: Provides detailed performance metrics, including confusion matrices and classification reports.
Model Persistence: Saves trained models and metadata for future use.
Visualization: Displays confusion matrices and performance graphs for better understanding.


⚕️Installation
Prerequisites
Make sure you have Python 3.x installed on your machine. You will also need the following packages:

pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn lime


⚕️Clone the Repository
git clone <https://github.com/Alex-Irae/DiseasePrediction>


⚕️Dataset
Ensure you have the dataset in the ressources folder named data.csv. The dataset should contain features representing symptoms and a target column named diagnostic representing the disease labels.

⚕️Usage
Run the script:

python PredictDis.py
Results: The script will read the dataset, preprocess the data, train the models, and save the best performing model as a .pkl file in the models directory.

Evaluation: Upon completion, the script will print out the accuracy of the models and display confusion matrices for better insights.

⚕️File Structure
		
	├── ressource
	│   ├── data.csv               	          # Input dataset
 	│   ├── disease_recommandation.json	  # Mapping of pgysician recommandation for each disease
	│   ├── disease_dict.json                 # Mapping of encoded disease labels (Generated at runtime)
	│   └── symptoms_list.json                # List of symptom features  (Generated at runtime)
	├── models                                # Directory for saved models
	├── ResultWindow.py                       # Display the results
	├── main.py                               # Create the UI
	├── PredictTrainer.py                     # Generate and train the models bbefore saving them, generates .json ressources files
	├── disease_predict.py                    # Use the models to calculate probabilities
	└── README.md                             # Project documentation


⚕️Contributing
If you would like to contribute to this project, feel free to submit a pull request or open an issue to discuss potential improvements.

⚕️License
This project is licensed under the MIT License. See the LICENSE file for more information.
