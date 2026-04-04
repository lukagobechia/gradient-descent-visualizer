import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config_loader import load_config

def train_test_split(X,y):
    config = load_config()
    test_size = config['data']['test_size']
    random_seed = config['data']['random_seed']

    np.random.seed(random_seed)

    n_samples = X.shape[0]

    # creating shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # calculating split point
    split = int(n_samples * (1 - test_size))

    # slicing into train and test
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def absorb_bias(X):
    n_samples = X.shape[0]
    bias = np.ones((n_samples, 1))
    return np.c_[bias, X]

def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, mean, std

def denormalize(X_norm, mean, std):
    return X_norm * std + mean

def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_norm, X_test_norm, mean, std = normalize(X_train, X_test)
    X_train_final = absorb_bias(X_train_norm)
    X_test_final = absorb_bias(X_test_norm)
    return X_train_final, X_test_final, y_train, y_test, mean, std

if __name__ == "__main__":
    from data.generate_data import generate_linear_data

    # generating fake data
    X, y = generate_linear_data()
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")

    # splitting it into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test  shape : {y_test.shape}")

    # normalizing the split data
    X_train_norm, X_test_norm, mean, std = normalize(X_train, X_test)
    print(f"\nX_train_norm shape : {X_train_norm.shape}")
    print(f"X_test_norm  shape : {X_test_norm.shape}")
    print(f"Mean : {mean}")
    print(f"Std : {std}")
