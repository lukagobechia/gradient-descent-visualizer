import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config_loader import load_config

def generate_linear_data():

    config = load_config()

    n_samples = config['data']['n_samples']
    noise = config['data']['noise']
    bias = config['data']['bias']
    weight = config['data']['weight']
    random_seed = config['data']['random_seed']

    np.random.seed(random_seed)

    X = np.linspace(-10, 20, n_samples)
    noise_values = np.random.normal(0, noise, n_samples)
    y = weight * X + bias + noise_values

    X = X.reshape(-1,1)

    return X, y

def plot_data(X, y):
    config = load_config()

    figsize = config['visualization']['figsize']
    save_plots = config['visualization']['save_plots']
    show_plots = config['visualization']['show_plots']

    plt.figure(figsize=figsize)
    plt.scatter(X,y, alpha=0.4, color="steelblue", label="fake data points")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Generated Linear Data')
    plt.legend()
    plt.tight_layout()

    if save_plots:
        plt.savefig('results/generated_data.png')

    if show_plots:
        plt.show()

if __name__ == "__main__":
    X, y = generate_linear_data()
    print("Generated data shapes: X:", X.shape, "y:", y.shape)
    plot_data(X, y)