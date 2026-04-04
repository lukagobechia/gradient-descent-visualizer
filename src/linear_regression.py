import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class LinearRegression:

    def __init__(self):
        self.w = None
        self.b = None
        self.loss_history = []


    def initialize_weights(self, n_features):
        self.w = np.random.randn(n_features)
        self.b = 0.0

    def predict(self, X):
        y_hat = X@self.w + self.b
        return y_hat

    def mse(self, y, y_hat):
        n = y.shape[0]
        error = y_hat - y
        mse = np.sum(error**2) / (n * 2)
        return mse

    def rmse(self, y, y_hat):
        mse = self.mse(y, y_hat)
        rmse = np.sqrt(mse)
        return rmse

    def mae(self, y, y_hat):
        error = y_hat - y
        mae = np.mean(np.abs(error))
        return mae

    def r2_score(self, y, y_hat):
    
        ss_res = np.sum((y_hat - y) ** 2)
        ss_tot = np.sum((y-y.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def evaluate(self, X, y):
        y_hat = self.predict(X)
        mse = self.mse(y, y_hat)
        rmse = self.rmse(y, y_hat)
        mae = self.mae(y, y_hat)
        r2 = self.r2_score(y, y_hat)
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        "r2": r2
    }
