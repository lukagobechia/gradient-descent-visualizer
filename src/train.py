import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.generate_data import generate_linear_data
from src.utils import preprocess
from src.optimizers.batch_gd import BatchGD
from src.optimizers.stochastic_gd import StochasticGD
from src.optimizers.minibatch_gd import MiniBatchGD


def train_all_models():
    X, y = generate_linear_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)

    print("Training Batch GD...")
    batch_model = BatchGD()
    batch_model.fit(X_train, y_train)

    print("Training Stochastic GD...")
    sgd_model = StochasticGD()
    sgd_model.fit(X_train, y_train)

    print("Training Mini-Batch GD...")
    minibatch_model = MiniBatchGD()
    minibatch_model.fit(X_train, y_train)

    models = {
        "Batch GD"      : batch_model,
        "Stochastic GD" : sgd_model,
        "Mini-Batch GD" : minibatch_model,
    }

    return models, X_test, y_test