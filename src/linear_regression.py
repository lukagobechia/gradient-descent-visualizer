import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def predict(X, w, b):
    y_hat = X@w + b
    return y_hat

def mse(y, y_hat):
    n = y.shape[0]
    error = y_hat - y
    mse = np.sum(error**2) / (n * 2)
    return mse

def r2_score(y, y_hat):
 
    ss_res = np.sum((y_hat - y) ** 2)
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def initialize_weights(n_features):
    w = np.random.randn(n_features)
    b = 0.0
    return w, b

if __name__ == "__main__":
    from data.generate_data import generate_linear_data
    from src.utils import train_test_split, normalize

    # generating fake data
    X, y = generate_linear_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_norm, X_test_norm, mean, std = normalize(X_train, X_test)

    n_features = X_train_norm.shape[1]
    w, b = initialize_weights(n_features)
    print(f"Initial weights : w={w}, b={b}")

    y_hat = predict(X_train_norm, w, b)
    print(f"y_hat shape     : {y_hat.shape}")
    print(f"y_hat first 5   : {y_hat[:5]}")

    loss = mse(y_train, y_hat)
    r2 = r2_score(y_train, y_hat)
    print(f"\nUntrained model:")
    print(f"  MSE : {loss:.4f}")
    print(f"  R²  : {r2:.4f}")