# Linear Regression Implementation in Python

This Python script implements a basic Linear Regression model from scratch using NumPy and Pandas. It includes both simple and polynomial regression capabilities, as well as a method for regularizing the polynomial regression to prevent overfitting. The script also demonstrates how to normalize features, split data into training and test sets, and evaluate model performance using R-squared and Mean Absolute Error (MAE) metrics.

## Features

- Simple Linear Regression: Estimate the relationship between two continuous variables.
- Polynomial Regression: Extend the model to capture non-linear relationships between variables.
- Regularization: Include a regularization term to the polynomial regression to control overfitting.
- Data Normalization: Standardize the feature set to have zero mean and unit variance.
- Model Evaluation: Compute R-squared and MAE to assess the model's performance.

## Requirements

- Python
- NumPy
- Pandas


## Usage

1. Data Preparation: The script expects a CSV file named `data.csv` with the dataset. The file should include both the features and the target variable.
   
2. Feature Selection: Specify the feature columns and the target variable column in the script. By default, it uses generic placeholder names.

3. Model Training and Evaluation:
   - For simple linear regression, instantiate the LinearRegression class without parameters.
   - For polynomial regression, instantiate with the desired degree of polynomial features and the regularization lambda if needed.
