import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.linear_regression import LinearRegression
from src.config_loader import load_config
class BatchGD(LinearRegression):
    def __init__(self):
        super().__init__()
        self.learning_rate = None
        self.epochs = None

    def fit(self, X, y):
        config = load_config()
        self.learning_rate = config['model']['learning_rate']
        self.epochs = config['model']['epochs']
        tolerance = config['model']['tolerance']

        n_samples,n_features = X.shape
        self.initialize_weights(n_features)

        for epoch in range(self.epochs):
            y_hat = self.predict(X)
            error = y_hat - y
            gradient = (1/n_samples) * X.T @error

            self.w = self.w - self.learning_rate * gradient

            loss = self.mse(y, y_hat)
            self.loss_history.append(loss)

            if epoch > 0:
                loss_change = abs(self.loss_history[-2] - self.loss_history[-1])
                if loss_change < tolerance:
                    print(f"Converged at epoch {epoch}")
                    break

            if epoch % 100 == 0:
                print(f"epoch {epoch:4d} loss: {loss:.4f}")
        return self