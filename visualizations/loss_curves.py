import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.generate_data import generate_linear_data
from src.config_loader import load_config
from src.optimizers.batch_gd import BatchGD
from src.optimizers.minibatch_gd import MiniBatchGD
from src.optimizers.stochastic_gd import StochasticGD
from src.utils import preprocess


def plot_loss_curves():
    config = load_config()

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

    batch_loss = batch_model.loss_history
    sgd_loss = sgd_model.loss_history
    minibatch_loss = minibatch_model.loss_history

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    models = [
        (batch_loss, "Batch GD", "steelblue"),
        (sgd_loss, "Stochastic GD", "tomato"),
        (minibatch_loss, "Mini-Batch GD", "mediumseagreen"),
    ]

    for ax, (loss, name, color) in zip(axes, models):
        ax.plot(loss, color=color, linewidth=2)
        ax.set_title(f"{name}\n final loss: {loss[-1]:.2f} | epochs: {len(loss)}")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Loss Curves — Batch vs SGD vs Mini-Batch", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    # Step 5 — save and show
    save_plots = config["visualization"]["save_plots"]
    show_plots = config["visualization"]["show_plots"]

    if save_plots:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/loss_curves.png", dpi=150, bbox_inches="tight")
        print("saved to results/loss_curves.png")

    if show_plots:
        plt.show()


if __name__ == "__main__":
    plot_loss_curves()
