import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.generate_data import generate_linear_data
from src.utils import preprocess
from src.linear_regression import LinearRegression
from src.optimizers.batch_gd import BatchGD


class Test:
    def __init__ (self):
        pass

    def test_linear_regression(self):
        X, y                          = generate_linear_data()
        X_train, _, y_train, _, _, _  = preprocess(X, y)

        model = LinearRegression()
        model.initialize_weights(X_train.shape[1])

        print("--------START of Linear Regresion--------")

        print(f"w     : {model.w}")

        metrics = model.evaluate(X_train, y_train)
        for name, val in metrics.items():
            print(f"{name} : {val:.4f}")

        print("--------END of Linear Regresion--------")

    def test_batchGD(self):
        X, y = generate_linear_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)

        # train
        model = BatchGD()
        model.fit(X_train, y_train)

        print("--------START of Batch Gradient Descent--------")

        # results
        print(f"\n--- Results ---")
        print(f"w : {model.w}")             # w[0]=bias, w[1]=weight

        metrics = model.evaluate(X_test, y_test)
        for name, val in metrics.items():
            print(f"{name:5s} : {val:.4f}")

        print("--------END of Batch Gradient Descent--------")


if __name__ == "__main__":
    t = Test()
    t.test_linear_regression()
    t.test_batchGD()