import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.linear_regression import LinearRegression
from src.config_loader import load_config

class StochasticGD(LinearRegression):
    def __init__(self):
        super().__init__()
        self.learning_rate = None
        self.epochs = None

    def fit(self,X,y):
        config = load_config()

        self.learning_rate = config['model']['learning_rate']
        self.epochs = config['model']['epochs']
        tolerance = config['model']['tolerance']

        n_samples, n_features = X.shape

        self.initialize_weights(n_features)
        y_hat = self.predict(X)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            for i in indices:
                X_i = X[i].reshape(1,-1) # (1, 2) -> (1,2) @ (2,) to be possible
                y_hat_i = self.predict(X_i)
                error = y_hat_i - y[i].squeeze()

                gradient = (X_i.T * error).flatten()

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