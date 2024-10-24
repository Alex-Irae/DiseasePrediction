// Disclaimer\\\

This is a self-taught project and should not in any case be taken as medical advice, if you happen to be sick, check with a specialist.

⚕️Disease Prediction System 🏥.

⚕️Overview.

This project implements a machine learning-based system to predict diseases based on a list of symptoms. It utilizes various classification algorithms, including SVM, Random Forest, XGBoost, and Logistic Regression, and a Depp Neural Network (SIgmoid,ReLu,Swish,SoftMax) to analyze the input data and provide accurate predictions. The system also incorporates techniques for data balancing, hyperparameter tuning, and model evaluation.


⚕️Features.

Data Preprocessing: Reads data from CSV, handles missing values, and encodes categorical labels.
Data Balancing: Utilizes SMOTE to oversample minority classes and create a balanced dataset.
Model Training: Trains multiple models using cross-validation and selects the best performing model.
DNN Training : Runs Hyperparameter tuning using a grid search and optuna for best layer size/epochs/learning rate combination
Hyperparameter Tuning: Optimizes model parameters using GridSearchCV for improved accuracy.
Model Evaluation: Provides detailed performance metrics, including confusion matrices and classification reports.
Model Persistence: Saves trained models and metadata for future use.
Visualization: Displays confusion matrices and performance graphs for better understanding.


⚕️Installation.

Prerequisites.
Make sure you have Python 3.10 installed on your machine. You will also need the following packages:

pip install numpy PyQt5 pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn tf(for gpu conmputing)

⚕️Recommandations.

Some models couldn't be loaded onto the project due to thier sizes, run once the file MlTrainer.py to get all the models (the run is relatively long due to xgboost pipeline (comment if not needed))
For the cnn model, feel free to play around with the hyperparameters, I upoloaded 2 models with good accuracy and good generalization (96%)

⚕️Clone the Repository.

git clone <https://github.com/Alex-Irae/DiseasePrediction>

⚕️Dataset.

Ensure you have the dataset in the ressources folder named data.csv. The dataset should contain features representing symptoms and a target column named diagnostic representing the disease labels.

⚕️Usage.

Run the script.

python main.py.

Results: an UI will be displayed asking to enter symptoms based on real diseases, 
Once entered, press the subit button to see the predicted disease. 2 new windows will be displayed, one with the results for the neural network and one with different machine learning models.
The accuracy of the Neural network is higher on unseen data and random inputs.
On the result window, you can add the sample to the data set and compute the models and neural network once more  with the add data button.
with the check recommandation button, you can check the recommandation from a professional if you happen to haave this particular disease
with the symptom influence button, you cna check which symptom influenced the most for each model (some models dont dispose of this function)


⚕️File Structure.

		
	├── ressource
	│   ├── data.csv               	          # Input dataset
 	│   ├── disease_recommandation.json	  # Mapping of pgysician recommandation for each disease
	│   ├── disease_dict.json                 # Mapping of encoded disease labels (Generated at runtime)
	│   └── symptoms_list.json                # List of symptom features  (Generated at runtime)
	├── models                                # Directory for saved models
	├── Ml_Result_Window.py                   # Display the results for the Ml models
 	├── Cnn_Result_window			  # Display the rsults for the choosen cnn model
  	├── Cnn.py				  # Class for the neural network, initialisation and compute
   	├── CnnTrainer.py			  # File to process the data and train the Neural network
	├── main.py                               # Create the UI
	├── MlTrainer.py                          # Generate and train the models bbefore saving them, generates .json ressources files
	├── utils.py                    	  # Use the models to calculate probabilities
	└── README.md                             # Project documentation


⚕️Contributing.

If you would like to contribute to this project, feel free to submit a pull request or open an issue to discuss potential improvements.

⚕️License.

This project is licensed under the MIT License. See the LICENSE file for more information.


