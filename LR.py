# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("data.csv")

# %%
class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b.dot(self.coefficients)

def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - numerator / denominator

data = pd.read_csv('data.csv')
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
X = data[features].values
y = data['price'].values

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# %%
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

lin_reg.fit(X_train_normalized, y_train)

predictions = lin_reg.predict(X_test_normalized)
print("R^2 Score:", r2_score(y_test, predictions))

# %% [markdown]
# # Adding polynomial features

# %%
class LinearRegression:
    def __init__(self, regularization_lambda=0, degree=1):
        self.coefficients = None
        self.regularization_lambda = regularization_lambda
        self.degree = degree

    def _add_polynomial_features(self, X):
        X_poly = X
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X, y):
        X_poly = self._add_polynomial_features(X)
        X_b = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
        reg_identity = np.eye(X_b.shape[1])
        reg_identity[0, 0] = 0 
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b) + self.regularization_lambda * reg_identity).dot(X_b.T).dot(y)

    def predict(self, X):
        X_poly = self._add_polynomial_features(X)
        X_b = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
        return X_b.dot(self.coefficients)


# %%
degree = 5
regularization_lambda = 0.007

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

lin_reg = LinearRegression(regularization_lambda=regularization_lambda, degree=degree)
lin_reg.fit(X_train_normalized, y_train)

predictions = lin_reg.predict(X_test_normalized)
print("R^2 Score:", r2_score(y_test, predictions))


# %%

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

y_true = y_test
y_pred = predictions

print("R^2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error (MAE):", mae(y_true, y_pred))


