import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.generate_data import generate_linear_data
from src.config_loader import load_config
from src.train import train_all_models
from src.utils import preprocess


def compute_loss_surface(X, y, w0_range, w1_range):
    W0, W1 = np.meshgrid(w0_range, w1_range)
    losses = np.zeros_like(W0)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w = np.array([W0[i, j], W1[i, j]])
            y_hat = X @ w
            losses[i, j] = np.mean((y_hat - y) ** 2)

    return W0, W1, losses


def plot_weight_trajectory():
    config = load_config()
    models, _, _ = train_all_models()

    batch_model = models["Batch GD"]
    sgd_model = models["Stochastic GD"]
    minibatch_model = models["Mini-Batch GD"]

    w_final = batch_model.w
    print(f"w_final: {w_final}")

    w0_range = np.linspace(0, w_final[0] + 20, 100)
    w1_range = np.linspace(0, w_final[1] + 10, 100)

    X, y = generate_linear_data()
    X_train, _, y_train, _, _, _ = preprocess(X, y)

    W0, W1, losses = compute_loss_surface(X_train, y_train, w0_range, w1_range)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    optimizer_list = [
        (batch_model, "Batch GD", "steelblue"),
        (sgd_model, "Stochastic GD", "tomato"),
        (minibatch_model, "Mini-Batch GD", "mediumseagreen"),
    ]

    for ax, (model, name, color) in zip(axes, optimizer_list):
        ax.contourf(W0, W1, losses, levels=50, cmap="YlOrRd_r", alpha=0.8)
        ax.contour(W0, W1, losses, levels=20, colors="white", linewidths=0.5, alpha=0.8)

        history = np.array(model.weight_history)

        ax.plot(
            history[:, 0],
            history[:, 1],
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label="path",
        )

        ax.scatter(
            history[0, 0], history[0, 1], color="white", s=80, zorder=5, label="start"
        )

        ax.scatter(
            history[-1, 0],
            history[-1, 1],
            color="yellow",
            s=150,
            marker="*",
            zorder=5,
            label="end",
        )

        ax.set_title(name)
        ax.set_xlabel("w₀ (bias)")
        ax.set_ylabel("w₁ (weight)")
        ax.legend(fontsize=8)

    plt.suptitle("Weight Trajectory on Loss Surface", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_plots = config["visualization"]["save_plots"]
    show_plots = config["visualization"]["show_plots"]

    if save_plots:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/weight_trajectory.png", dpi=150, bbox_inches="tight")
        print("Saved to results/weight_trajectory.png")

    if show_plots:
        plt.show()


if __name__ == "__main__":
    plot_weight_trajectory()
