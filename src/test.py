import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.generate_data import generate_linear_data
from src.utils import preprocess
from src.linear_regression import LinearRegression
from src.optimizers.batch_gd import BatchGD
from src.optimizers.stochastic_gd import StochasticGD
from src.optimizers.minibatch_gd import MiniBatchGD


class Test:
    def __init__ (self):
        pass

    def test_linear_regression(self):
        print("=" * 40)
        print("--------START of Linear Regresion--------")
        print("=" * 40)

        X, y                          = generate_linear_data()
        X_train, _, y_train, _, _, _  = preprocess(X, y)

        model = LinearRegression()
        model.initialize_weights(X_train.shape[1])

        print(f"w     : {model.w}")

        metrics = model.evaluate(X_train, y_train)
        for name, val in metrics.items():
            print(f"{name} : {val:.4f}")

    def test_batchGD(self):
        print("=" * 40)
        print("--------START of Batch Gradient Descent--------")
        print("=" * 40)

        X, y = generate_linear_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)
        
        # train
        model = BatchGD()
        model.fit(X_train, y_train)

        # results
        print(f"\n--- Results ---")
        print(f"w : {model.w}")

        metrics = model.evaluate(X_test, y_test)
        for name, val in metrics.items():
            print(f"{name:5s} : {val:.4f}")

    def test_stochasticGD(self):
        print("=" * 40)
        print("--------START of Stochastic Gradient Descent--------")
        print("=" * 40)

        X, y = generate_linear_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)

        model = StochasticGD()
        model.fit(X_train, y_train)

        print(f"\n--- Results ---")
        print(f"w : {model.w}")

        metrics = model.evaluate(X_test, y_test)
        for name, val in metrics.items():
            print(f"{name:5s} : {val:.4f}")
            
    def test_miniBatchGD(self):
        print("=" * 40)
        print("--------START of Mini-Batch Gradient Descent--------")
        print("=" * 40)

        X, y = generate_linear_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)

        model = MiniBatchGD()
        model.fit(X_train, y_train)

        print(f"\n--- Results ---")
        print(f"w : {model.w}")

        metrics = model.evaluate(X_test, y_test)
        for name, val in metrics.items():
            print(f"{name:5s} : {val:.4f}")

if __name__ == "__main__":
    t = Test()
    t.test_linear_regression()
    t.test_batchGD()
    t.test_stochasticGD()
    t.test_miniBatchGD()