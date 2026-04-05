# gradient-descent-visualizer

> Implementing and comparing Batch, Stochastic, and Mini-Batch
> Gradient Descent from scratch using NumPy, no scikit-learn.

---

## 📌 What This Project Shows

- How gradient descent learns by adjusting weights step by step
- The difference between Batch, Stochastic, and Mini-Batch optimizers
- How the bias trick absorbs the intercept into the weight vector
- How normalization affects training speed and stability
- Evaluation of models using MSE, RMSE, MAE, and R² from scratch

---

## 🧮 Math Overview

### Linear Regression

The model predicts output as:

$$\hat{y} = Xw$$

Where $X$ includes a bias column of ones so no separate bias term $b$ is needed:

$$X = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}, \quad w = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$$

---

### Loss Function — Mean Squared Error

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

---

### Gradient Derivation

Taking the derivative of MSE with respect to $w$:

$$\frac{\partial MSE}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)$$

### Weight Update Rule

$$w := w - \alpha \cdot \frac{\partial MSE}{\partial w}$$

Where $\alpha$ is the learning rate.

---

### The Bias Trick

Instead of computing $\hat{y} = Xw + b$ separately, a column of ones is prepended to $X$:

$$X_{bias} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}$$

This absorbs $b$ into $w_0$, so the update rule becomes a single matrix operation.

---

### The Three Optimizers

| Method | Samples per update | Updates per epoch | Gradient |
|---|---|---|---|
| Batch GD | all $n$ | $1$ | $\frac{1}{n} X^T(\hat{y} - y)$ |
| Stochastic GD | $1$ | $n$ | $X_i^T(\hat{y}_i - y_i)$ |
| Mini-Batch GD | $b$ | $n/b$ | $\frac{1}{b} X_b^T(\hat{y}_b - y_b)$ |

---

### Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MSE | $\frac{1}{n}\sum(\hat{y} - y)^2$ | lower is better |
| RMSE | $\sqrt{MSE}$ | same unit as $y$ |
| MAE | $\frac{1}{n}\sum\|\hat{y} - y\|$ | robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 1.0 = perfect fit |

---

## 📁 Project Structure

```
gradient-descent-visualizer/
│
│
├── data/
│   └── generate_data.py              ← synthetic dataset generator
│
├── notebooks/
│   └── demo.ipynb                    ← interactive walkthrough
│
├── results/
├── comparison_table.csv
├── comparison_table.png
├── generated_data.png
├── loss_curves.png
├── weight_trajectory.png
│
├── src/
│   ├── config_loader.py              ← loads config.yml
│   ├── linear_regression.py          ← base class: predict, mse, rmse, mae, r2
│   ├── utils.py                      ← split, normalize, absorb_bias, preprocess
│   ├── train.py                      ← train all 3 models
│   ├── test.py                       ← tests for all models
│   └── optimizers/
│       ├── batch_gd.py               ← batch gradient descent
│       ├── minibatch_gd.py           ← mini-batch gradient descent
│       └── stochastic_gd.py          ← stochastic gradient descent
│
├── visualizations/
│   ├── comparison_table.py           ← a summary table comparing all 3 optimizers
│   ├── loss_curves.py                ← loss vs epoch for all 3 optimizers
│   └── weight_trajectory.py          ← weight path on loss surface contour
│
├── .gitignore
├── config.yaml                        ← all hyperparameters
├── README.md
├── requirements.txt
├── ruff.toml

```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/lukagobechia/gradient-descent-visualizer.git
cd gradient-descent-visualizer
```

### 2. Create and activate environment
```bash
conda create -n ml python=3.10
conda activate ml
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate data and verify
```bash
python data/generate_data.py
```

### 5. Run all optimizer tests
```bash
python src/test.py
```
---

## ⚙️ Configuration

All hyperparameters live in `config.yml`, no need to touch Python files:

```yaml
data:
  n_samples: 500
  weight: 2.8
  bias: 30.0
  noise: 3.0
  test_size: 0.2
  random_seed: 42

model:
  learning_rate: 0.01
  epochs: 1000
  tolerance: 1.0e-6
  mini_batch:
    batch_size: 32
```

---

## 📊 Results

> All models trained on 500 synthetic data points: **Study Hours vs Exam Score**
> (`weight=2.8`, `bias=30.0`, `noise=3.0`, `random_seed=42`)

---

### 📉 Loss Curves

Tracks MSE loss at every epoch for all three optimizers.

![loss curves](results/loss_curves.png)

| Optimizer | Behavior |
|---|---|
| **Batch GD** | Smooth and stable. It uses all 400 samples every epoch |
| **Stochastic GD** | Converges in epoch 1 but stays noisy. 400 updates per epoch |
| **Mini-Batch GD** | Fast and stable. It converges in ~85 epochs using 32 samples per step |

---

### 🗺️ Weight Trajectory on Loss Surface

Shows the path each optimizer's weights took during training.
Background color represents MSE, **darker = lower loss**, center = minimum.

![weight trajectory](results/weight_trajectory.png)

| Optimizer | Path |
|---|---|
| **Batch GD** | Smooth straight path from `[0,0]` to minimum |
| **Stochastic GD** | Converges instantly, start and end is nearly identical |
| **Mini-Batch GD** | Slightly noisy but direct path to minimum |

---

### 📊 Evaluation Table

All models evaluated on held-out test set (20% of 500 samples = 100 samples).

![comparison table](results/comparison_table.png)

**Key takeaways from the numbers:**
- All three optimizers achieve similar **R² ≈ 0.98** same final quality
- **Mini-Batch GD** converges fastest in fewest epochs
- **Stochastic GD** is slowest in wall-clock time due to 400 updates per epoch
- **Batch GD** is most stable but takes the most epochs to converge
---

## 🔍 Key Takeaways

- **Batch GD** is stable and smooth but loads all data at once so it is not ideal for large datasets
- **Stochastic GD** updates weights after every single sample so ir is memory efficient but noisy convergence
- **Mini-Batch GD** is the best of both worlds, stable like Batch GD and fast like SGD. It is most used in practice
- The **bias trick** simplifies the math by absorbing the intercept into the weight vector. one update rule covers everything
- **Normalization must happen before** adding the bias column, otherwise the ones column gets incorrectly scaled
- All three optimizers converge to the same weights, the difference is **speed**, **memory usage**, and **noise**

---

## 📚 References
- NumPy Documentation — https://numpy.org/doc/