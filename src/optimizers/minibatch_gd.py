import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.linear_regression import LinearRegression
from src.config_loader import load_config

class MiniBatchGD(LinearRegression):
    def __init__(self):
        super().__init__()
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None

    def fit(self, X, y):
        config = load_config()

        self.learning_rate = config['model']['learning_rate']
        self.epochs = config['model']['epochs']
        self.batch_size = config['model']['mini_batch']['batch_size']
        tolerance = config['model']['tolerance']

        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):

                end = start + self.batch_size
                X_batch = X_shuffled[start:end]  
                y_batch = y_shuffled[start:end] 

                y_hat_batch = self.predict(X_batch)

                error = y_hat_batch - y_batch

                gradient = (X_batch.T @ error) * (1/X_batch.shape[0])

                self.w = self.w - self.learning_rate * gradient

            
            y_hat = self.predict(X)
            loss = self.mse(y, y_hat)
            self.loss_history.append(loss)

            if epoch > 0:
                loss_change = abs(self.loss_history[-2] - self.loss_history[-1])
                if loss_change < tolerance:
                    print(f"Converged at epoch {epoch}")
                    break

            if epoch % 100 == 0:
                print(f"epoch {epoch:4d}  loss: {loss:.4f}")
                
        return self


