import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.generate_data import generate_linear_data
from src.optimizers.batch_gd import BatchGD
from src.optimizers.minibatch_gd import MiniBatchGD
from src.optimizers.stochastic_gd import StochasticGD
from src.utils import preprocess


def run_comparison():
    X, y = generate_linear_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(X, y)

    optimizers = [
        ("Batch GD", BatchGD()),
        ("Stochastic GD", StochasticGD()),
        ("Mini-Batch GD", MiniBatchGD()),
    ]

    results = {}

    for name, model in optimizers:
        print(f"Training {name}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        elapsed = end_time - start_time

        metrics = model.evaluate(X_test, y_test)

        results[name] = {
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "epochs": len(model.loss_history),
            "time": elapsed,
        }

    print_table(results)
    save_results(results)
    save_table_as_png(results)

    return results


def print_table(results):
    print("\n")
    print("=" * 70)
    print(
        f"{'Optimizer':<15} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Epochs':>8} {'Time(s)':>8}"
    )
    print("=" * 70)

    for name, m in results.items():
        print(
            f"{name:<15} "
            f"{m['mse']:>8.4f} "
            f"{m['rmse']:>8.4f} "
            f"{m['mae']:>8.4f} "
            f"{m['r2']:>8.4f} "
            f"{m['epochs']:>8} "
            f"{m['time']:>8.4f}"
        )

    print("=" * 70)

    best_r2 = max(results, key=lambda x: results[x]["r2"])
    best_time = min(results, key=lambda x: results[x]["time"])
    best_epochs = min(results, key=lambda x: results[x]["epochs"])

    print(f"\n🏆 Best R²     : {best_r2}")
    print(f"⚡ Fastest     : {best_time}")
    print(f"📉 Fewest epochs: {best_epochs}")


def save_results(results):
    os.makedirs("results", exist_ok=True)

    with open("results/comparison_table.csv", "w") as f:
        f.write("Optimizer,MSE,RMSE,MAE,R2,Epochs,Time\n")

        for name, m in results.items():
            f.write(
                f"{name},"
                f"{m['mse']:.4f},"
                f"{m['rmse']:.4f},"
                f"{m['mae']:.4f},"
                f"{m['r2']:.4f},"
                f"{m['epochs']},"
                f"{m['time']:.4f}\n"
            )

    print("\nSaved to results/comparison_table.csv")


def save_table_as_png(results):
    columns = ["Optimizer", "MSE", "RMSE", "MAE", "R²", "Epochs", "Time(s)"]

    rows = []
    for name, m in results.items():
        rows.append(
            [
                name,
                f"{m['mse']:.4f}",
                f"{m['rmse']:.4f}",
                f"{m['mae']:.4f}",
                f"{m['r2']:.4f}",
                f"{m['epochs']}",
                f"{m['time']:.4f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    for col in range(len(columns)):
        table[0, col].set_facecolor("#2c3e50")
        table[0, col].set_text_props(color="white", fontweight="bold")

    for row in range(1, len(rows) + 1):
        color = "#f2f2f2" if row % 2 == 0 else "white"
        for col in range(len(columns)):
            table[row, col].set_facecolor(color)

    r2_values = [float(r[4]) for r in rows]
    best_r2_row = r2_values.index(max(r2_values)) + 1
    for col in range(len(columns)):
        table[best_r2_row, col].set_facecolor("#d4edda")

    plt.title(
        "Optimizer Comparison — Study Hours vs Exam Score",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison_table.png", dpi=150, bbox_inches="tight")
    print("Saved to results/comparison_table.png")
    plt.show()


if __name__ == "__main__":
    run_comparison()
