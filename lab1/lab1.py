import numpy as np
from sklearn.datasets import load_iris

def prepare_iris(seed, n_train_per_class=35):
    iris = load_iris()
    X, y = iris.data, iris.target
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for classes in range(3):
        idx = np.where(y == classes)[0]
        rng.shuffle(idx)
        train_idx.append(idx[:n_train_per_class])
        test_idx.append(idx[n_train_per_class:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, mean, std

def init_params(seed):
    rng = np.random.default_rng(seed)

    std1 = np.sqrt(2 / 4)
    W1 = rng.normal(loc=0.0, scale=std1, size=(4, 8))
    b1 = np.zeros(8)

    std2 = np.sqrt(2 / (8 + 3))
    W2 = rng.normal(loc=0.0, scale=std2, size=(8, 3))
    b2 = np.zeros(3)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_pass(params, X, y):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0)

    z2 = a1 @ W2 + b2
    shifted = z2 - np.max(z2, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probs = np.exp(log_probs)

    N = X.shape[0]
    loss = -np.mean(log_probs[np.arange(N), y])

    cache = {"X": X, "z1": z1, "a1": a1, "probs": probs, "y": y}
    return loss, cache

def backward_pass(params, cache, divide_by_n=True):
    X, z1, a1, probs, y = cache["X"], cache["z1"], cache["a1"], cache["probs"], cache["y"]
    W2 = params["W2"]
    N = X.shape[0]

    # dL/dz2 = (softmax - one_hot)
    dz2 = probs.copy()
    dz2[np.arange(N), y] -= 1
    if divide_by_n:
        dz2 /= N

    dW2 = a1.T @ dz2        # (8, 3)
    db2 = dz2.sum(axis=0)   # (3,)

    da1 = dz2 @ W2.T        # (N, 8)
    dz1 = da1 * (z1 > 0)    # ReLU derivative, (N, 8)

    dW1 = X.T @ dz1         # (4, 8)
    db1 = dz1.sum(axis=0)   # (8,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, mean, std = prepare_iris(seed=0)
    params = init_params(seed=0)
    loss, cache = forward_pass(params, X_train, y_train)
    grads = backward_pass(params, cache)
    print(f"loss: {float(loss)!r}")
    for name, g in grads.items():
        print(f"d{name}: shape {g.shape}")
