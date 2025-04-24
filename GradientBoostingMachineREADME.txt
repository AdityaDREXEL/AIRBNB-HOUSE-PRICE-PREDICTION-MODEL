# Gradient Boosting Machine (GBM) Implementation in Python

This Python script implements a Gradient Boosting Machine (GBM) for regression tasks, building upon a custom implementation of decision trees. The GBM model is designed to sequentially correct the errors of previous models by fitting new models to the residual errors made by the preceding steps. This implementation includes features such as variable learning rates, decision tree depth control, and handling of different sample sizes for splits and leaves.

## Features

- Custom Decision Tree Regressor: Foundation for the GBM, capable of fitting to residual errors.
- Gradient Boosting Regressor: Sequentially builds an ensemble of decision trees to improve prediction accuracy.
- Configurable Hyperparameters: Control over the number of estimators, learning rate, max depth of trees, minimum samples for splits, and minimum samples per leaf.
- Feature Importance Evaluation: Through the aggregation of decision tree splits, gain insights on feature importance.
- Model Evaluation: Utilize R-squared and Mean Absolute Error (MAE) metrics for performance evaluation.
- Data Splitting: Includes a function to split data into training and test sets for model validation.

## Requirements

- Python
- NumPy
- Pandas

## Usage

The script is structured to first define the decision tree model, followed by the GBM model that leverages these trees. It demonstrates how to preprocess the data, configure the model parameters, train the model, and evaluate its performance on a test set.

1. Data Preparation: Ensure your dataset is in a CSV file. The script anticipates the data to include both feature columns and a target variable column.

2. Model Configuration: Before running the script, define your model's hyperparameters, including the number of estimators, learning rate, maximum depth, and others.

3. Training and Evaluation: The script splits the dataset into training and testing sets, trains the GBM model on the training set, and evaluates its performance on the test set.

