# %% [markdown]
# # Decision Trees

# %%
import pandas as pd
import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=15, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features if max_features is not None else float('inf')
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_split = self._get_best_split(X, y, num_samples, num_features)
            if best_split['value'] is not None:
                left_X, right_X, left_y, right_y = self._split(X, y, best_split['feature_index'], best_split['threshold'])
                left_subtree = self._build_tree(left_X, left_y, depth + 1)
                right_subtree = self._build_tree(right_X, right_y, depth + 1)
                return DecisionTreeNode(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)
        leaf_value = np.mean(y)
        return DecisionTreeNode(value=leaf_value)

    def _get_best_split(self, X, y, num_samples, num_features):
        best_split = {'feature_index': None, 'threshold': None, 'value': None}
        max_variance_reduction = -np.inf
        
        features = np.random.choice(range(num_features), min(self.max_features, num_features), replace=False)

        for feature_index in features:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left_y, right_y = self._split_dataset(y, feature_values, threshold)
                variance_reduction = self._calculate_variance_reduction(y, left_y, right_y)
    
                if variance_reduction > max_variance_reduction:
                    max_variance_reduction = variance_reduction
                    best_split['feature_index'] = feature_index
                    best_split['threshold'] = threshold
                    best_split['value'] = max_variance_reduction
        return best_split

    def _calculate_variance_reduction(self, y, left_y, right_y):
        total_variance = np.var(y)
        left_variance = np.var(left_y) if left_y.size > 0 else 0
        right_variance = np.var(right_y) if right_y.size > 0 else 0
        weight_left = len(left_y) / len(y)
        weight_right = len(right_y) / len(y)
        weighted_variance = weight_left * left_variance + weight_right * right_variance
        variance_reduction = total_variance - weighted_variance
        return variance_reduction

    def _split(self, X, y, feature_index, threshold):
        left_idx = np.where(X[:, feature_index] <= threshold)[0]
        right_idx = np.where(X[:, feature_index] > threshold)[0]
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    def _split_dataset(self, y, feature_values, threshold):
        left_idx = np.where(feature_values <= threshold)[0]
        right_idx = np.where(feature_values > threshold)[0]
        return y[left_idx], y[right_idx]
    
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X])
    
    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)


# %% [markdown]
# # GBM

# %%
class GradientBoostingRegressor:
    def __init__(self, n_estimators=40, learning_rate=0.1, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)
        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
            tree.fit(X, residuals)
            update = tree.predict(X)
            predictions += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

def train_test_split(X, y, test_size=0.1, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - numerator / denominator

data = pd.read_csv('data.csv')  # Change the file path
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
X = data[features].values
y = data['price'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

best_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 4,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
model = GradientBoostingRegressor(**best_params)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

print(f"R^2 Score: {score}")

# %%
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mae = mean_absolute_error(y_test, predictions)
print(f"R^2 Score: {score}")
print(f"Mean Absolute Error (MAE): {mae}")


# %%



