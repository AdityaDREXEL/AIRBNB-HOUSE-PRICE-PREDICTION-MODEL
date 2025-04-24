🏡 House Price Prediction using Machine Learning
This project implements two core machine learning models—Linear Regression and Gradient Boosting Machines (GBM)—to predict house sale prices using structured real estate data. It aims to provide accurate property valuations, enabling smarter decision-making for buyers, sellers, and agents.

📁 Project Structure
Report.pdf: Project documentation outlining objectives, methodology, and results.

GBM.py: Implementation of a custom Gradient Boosting Regressor built from scratch using decision trees.

LR.py: Implementation of simple and polynomial Linear Regression with optional regularization.

GradientBoostingMachineREADME.txt & LinearRegressionREADME.txt: Component-specific documentation.

⚙️ Features
🔹 Linear Regression
Implements both simple and polynomial regression.

Includes regularization to avoid overfitting.

Uses R² and MAE for evaluation.

🔹 Gradient Boosting Machine
Custom Decision Tree Regressor as the weak learner.

Configurable GBM with support for:

Variable learning rates

Adjustable depth and sample splitting

Ensemble model of decision trees

Model training using residual fitting (boosting).

Evaluation via R² and MAE.

🧪 Model Evaluation Metrics
R² Score: Indicates how well predictions approximate actual values.

Mean Absolute Error (MAE): Measures average absolute prediction error.

📊 Final Results:

Model	R² Score	MAE
Linear Regression	0.5880	154,007
GBM	0.5978	148,671
🧰 Requirements
Python 3.x

Pandas

NumPy

🚀 How to Use
Install Dependencies
Install necessary packages (if not already installed):

bash
Copy
Edit
pip install numpy pandas
Prepare Data
Place your dataset in the same directory and name it data.csv. Make sure it includes the following columns:

bash
Copy
Edit
['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
'yr_built', 'yr_renovated', 'price']
Run Models

For GBM:

bash
Copy
Edit
python GBM.py
For Linear Regression:

bash
Copy
Edit
python LR.py
📌 Notes
Models use randomized train-test splits. Adjust seed value or split size in the scripts for experimentation.

Polynomial regression in LR.py can be tuned using degree and regularization_lambda.

