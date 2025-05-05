

house_price_readme = """
# House Price Prediction using Machine Learning

This project implements two core machine learning models—Linear Regression and Gradient Boosting Machines (GBM)—to predict house sale prices using structured real estate data. It aims to provide accurate property valuations, enabling smarter decision-making for buyers, sellers, and agents.

---

## Project Structure

- `Report.pdf`: Project documentation outlining objectives, methodology, and results.
- `GBM.py`: Implementation of a custom Gradient Boosting Regressor built from scratch using decision trees.
- `LR.py`: Implementation of simple and polynomial Linear Regression with optional regularization.
- `GradientBoostingMachineREADME.txt` & `LinearRegressionREADME.txt`: Component-specific documentation.

---

## Features

### Linear Regression

- Supports both simple and polynomial regression.
- Includes regularization to reduce overfitting.
- Evaluated using R² and MAE metrics.

### Gradient Boosting Machine

- Uses a custom Decision Tree Regressor as the weak learner.
- Configurable GBM with support for:
  - Variable learning rates
  - Adjustable tree depth and sample splitting
- Ensemble of decision trees trained via residual fitting (boosting).
- Evaluation via R² and MAE.

---

## Model Evaluation Metrics

- **R² Score**: Indicates how well predictions approximate actual values.
- **Mean Absolute Error (MAE)**: Measures average absolute prediction error.

### Final Results

| Model             | R² Score | MAE      |
|------------------|----------|----------|
| Linear Regression| 0.7880   | 89,450  |
| GBM              | 0.8578   | 56,320  |

---

## Requirements

- Python 3.x
- Pandas
- NumPy

---

## How to Use

### 1. Install Dependencies

```bash
pip install numpy pandas
