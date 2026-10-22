import numpy as np
import pandas as pd

def confusion(y,s,t):
    pred = s >= t
    actual = y == 1
    tp = pred & actual
    fp = pred & ~actual
    fn = ~pred & actual
    tn = ~pred & ~actual
    return tp.sum(), fp.sum(), fn.sum(), tn.sum()

def cost(fn, fp):
    return 10*fn+fp

def candidates(y, s):
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    is_last = np.append(s_sorted[:-1] != s_sorted[1:], True)
    thresholds = np.concatenate(([np.inf], s_sorted[is_last]))
    tp = np.concatenate(([0], tp_cum[is_last]))
    fp = np.concatenate(([0], fp_cum[is_last]))
    p = int((y == 1).sum())
    n = len(y) - p
    fn = p - tp
    tn = n - fp
    return {"threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "c": cost(fn, fp)}

def best_threshold(cand):
    i = int(np.argmin(cand["c"]))
    return cand["threshold"][i], i

def brute_force(y, s):
    thresholds = np.concatenate(([np.inf], np.unique(s)[::-1]))
    rows = [confusion(y, s, t) for t in thresholds]
    tp, fp, fn, tn = (np.array(col) for col in zip(*rows))
    c = cost(fn, fp)
    t_star = thresholds[c == c.min()].max()
    return {"threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "c": c}, t_star

def check_set(name, y, s):
    fast = candidates(y, s)
    slow, t_slow = brute_force(y, s)
    for key in fast:
        assert np.array_equal(fast[key], slow[key]), f"{name}: {key} differs"
    t_fast, _ = best_threshold(fast)
    assert t_fast == t_slow, f"{name}: t* differs ({t_fast} vs {t_slow})"
    print(f"set {name}: all {len(fast['threshold'])} candidates match, t* = {t_fast}, C = {slow['c'].min()}")
    print(pd.DataFrame(slow).to_string(index=False))

SETS = {
    "A": (np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1]), np.array([0, 1, 0, 1, 0, 0])),
    "B": (np.array([0.9, 0.6, 0.3]), np.array([0, 0, 0])),
    "C": (np.array([0.8] + [0.5] * 11 + [0.2]), np.array([1, 1] + [0] * 10 + [0])),
}

def run_checks():
    for name, (s, y) in SETS.items():
        check_set(name, y, s)

if __name__ == "__main__":
    run_checks()
