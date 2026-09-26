import argparse

import numpy as np
import torch
from torch import nn

from lab1 import prepare_iris, init_params, forward_pass, backward_pass

TORCH_TOL = 1e-12
NUMERIC_TOL = 1e-7
EPS = 1e-6

def build_torch_model(params):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"]))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"]))
    return model

def torch_loss_and_grads(params, X, y):
    model = build_torch_model(params)
    logits = model(torch.from_numpy(X))
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(y))
    loss.backward()
    grads = {
        "W1": model[0].weight.grad.numpy().T,
        "b1": model[0].bias.grad.numpy(),
        "W2": model[2].weight.grad.numpy().T,
        "b2": model[2].bias.grad.numpy(),
    }
    return loss.item(), grads

def compare_with_torch(params, X, y, divide_by_n=True):
    loss_np, cache = forward_pass(params, X, y)
    grads_np = backward_pass(params, cache, divide_by_n)
    loss_pt, grads_pt = torch_loss_and_grads(params, X, y)

    rows = [("Втрата", abs(loss_np - loss_pt), np.isfinite([loss_np, loss_pt]).all())]
    for name in ("W1", "b1", "W2", "b2"):
        diff = np.abs(grads_np[name] - grads_pt[name]).max()
        finite = np.isfinite(grads_np[name]).all() and np.isfinite(grads_pt[name]).all()
        rows.append((f"Градієнт {name}", diff, finite))

    print(f"Втрата NumPy:   {float(loss_np)!r}")
    print(f"Втрата PyTorch: {float(loss_pt)!r}\n")
    print("| Величина | Максимальна абсолютна різниця NumPy / PyTorch | Перевірку пройдено |")
    print("|---|---|---|")
    for name, diff, finite in rows:
        ok = finite and diff <= TORCH_TOL
        print(f"| {name} | {diff:.3e} | {'так' if ok else 'ні'} |")

def numerical_derivative(params, X, y, name, idx):
    original = params[name][idx]
    params[name][idx] = original + EPS
    loss_plus, _ = forward_pass(params, X, y)
    params[name][idx] = original - EPS
    loss_minus, _ = forward_pass(params, X, y)
    params[name][idx] = original
    return (loss_plus - loss_minus) / (2 * EPS)

def numerical_check(params, X, y, divide_by_n=True):
    _, cache = forward_pass(params, X, y)
    grads = backward_pass(params, cache, divide_by_n)

    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("|---|---|---|---|---|")
    for name, idx in (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))):
        g_manual = grads[name][idx]
        g_num = numerical_derivative(params, X, y, name, idx)
        diff = abs(g_num - g_manual)
        ok = np.isfinite(diff) and diff <= NUMERIC_TOL
        label = f"{name}{list(idx)}"
        print(f"| {label} | {g_manual:.12f} | {g_num:.12f} | {diff:.3e} | {'так' if ok else 'ні'} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug", action="store_true",
                        help="прибрати ділення на N у градієнті за логітами (пункт 4)")
    args = parser.parse_args()
    divide_by_n = not args.bug

    X_train, X_test, y_train, y_test, mean, std = prepare_iris(seed=0)
    params = init_params(seed=0)

    mode = "з навмисною помилкою (без ділення на N)" if args.bug else "правильна"
    print(f"Реалізація backward: {mode}, N = {X_train.shape[0]}\n")
    print("## Звірка з PyTorch\n")
    compare_with_torch(params, X_train, y_train, divide_by_n)
    print("\n## Чисельна перевірка\n")
    numerical_check(params, X_train, y_train, divide_by_n)
