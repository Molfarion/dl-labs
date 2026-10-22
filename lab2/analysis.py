"""Аналіз зі збережених прогнозів: перевірки, пошук порога на validation, криві, оцінка на test."""
from lab2_starter import load_predictions
from threshold import confusion, cost, candidates, best_threshold, run_checks
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
OUT = Path(__file__).resolve().parent / "results"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#52514e", "#8a8984", "#e6e5e1"
STYLE = [("t = 0.5", "o", "#2a78d6"), ("t*", "s", "#eb6834"), ("t = +inf", "^", "#1baf7a")]

def operating_points(y, s, t_star):
    rows = []
    for (label, _, _), t in zip(STYLE, (0.5, t_star, np.inf)):
        tp, fp, fn, tn = (int(v) for v in confusion(y, s, t))
        rows.append({"rule": label, "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "c": int(cost(fn, fp))})
    return rows

def new_axes(xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)
    return fig, ax

def plot_pr(y, s, points, path):
    precision, recall, _ = precision_recall_curve(y, s)
    fig, ax = new_axes("Recall", "Precision", "Precision-recall, validation")
    ax.plot(recall, precision, color=INK, lw=2, label="PR curve")
    ax.axhline(y.mean(), color=MUTED, lw=1, ls="--", label=f"no skill (prevalence {y.mean():.4f})")
    for p, (label, marker, colour) in zip(points, STYLE):
        if p["tp"] + p["fp"] == 0:
            continue
        pr, rc = p["tp"] / (p["tp"] + p["fp"]), p["tp"] / (p["tp"] + p["fn"])
        ax.scatter(rc, pr, marker=marker, color=colour, s=70, edgecolor=SURFACE, lw=2, zorder=3,
                   label=f"{label}: precision {pr:.3f}, recall {rc:.3f}")
    ax.legend(loc="lower left", frameon=False)
    fig.text(0.01, 0.01, "t = +inf is not shown: no positive predictions, precision undefined.", color=INK, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_roc(y, s, points, path):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    fig, ax = new_axes("False positive rate", "True positive rate (recall)", f"ROC, validation (AUC = {auc:.4f})")
    ax.plot(fpr, tpr, color=INK, lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="no skill")
    for p, (label, marker, colour) in zip(points, STYLE):
        x, yy = p["fp"] / (p["fp"] + p["tn"]), p["tp"] / (p["tp"] + p["fn"])
        ax.scatter(x, yy, marker=marker, color=colour, s=70, edgecolor=SURFACE, lw=2, zorder=3,
                   label=f"{label}: FPR {x:.4f}, TPR {yy:.3f}")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return auc

def ratio(num, den):
    return num / den if den else None

def test_evaluation(res, t):
    y, s = res["test_y"], res["test_scores"]
    tp, fp, fn, tn = (int(v) for v in confusion(y, s, t))
    return {"threshold": float(t), "n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "c": int(cost(fn, fp)), "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}

def error_table(res, t, k=5):
    """First k false positives and first k false negatives by CSV row number."""
    y, s = res["test_y"], res["test_scores"]
    rows, amount = res["test_row_ids"], res["test_amount"]
    pred = s >= t
    parts = []
    for kind, mask in (("FP", pred & (y == 0)), ("FN", ~pred & (y == 1))):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(rows[idx])][:k]
        parts.append(pd.DataFrame({"csv_row": rows[idx], "type": kind, "label": y[idx],
                                   "score": s[idx], "amount": amount[idx], "score_minus_t": s[idx] - t}))
    return pd.concat(parts, ignore_index=True)

def main():
    run_checks()   # test is touched only after all three checks pass
    res = load_predictions(OUT)
    yv, sv = res["validation_y"], res["validation_scores"]
    cand = candidates(yv, sv)
    t_star, _ = best_threshold(cand)
    pd.DataFrame(cand).to_csv(OUT / "candidates.csv", index=False)
    (OUT / "threshold.json").write_text(json.dumps({"t_star": float(t_star)}) + "\n")
    points = operating_points(yv, sv, t_star)
    pd.DataFrame(points).to_csv(OUT / "validation_comparison.csv", index=False)
    plot_pr(yv, sv, points, OUT / "pr_curve.png")
    auc = plot_roc(yv, sv, points, OUT / "roc_curve.png")
    print(f"t* = {t_star!r}, ROC-AUC = {auc:.6f}")
    print(pd.DataFrame(points).to_string(index=False))

    # test: threshold is read from disk and never tuned here
    t_fixed = json.loads((OUT / "threshold.json").read_text())["t_star"]
    result = test_evaluation(res, t_fixed)
    (OUT / "test_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    errors = error_table(res, t_fixed)
    errors.to_csv(OUT / "test_errors.csv", index=False)
    print(json.dumps(result, indent=2))
    print(errors.to_string(index=False))
    print(f"median Amount, whole test set: {np.median(res['test_amount']):.2f}")

if __name__ == "__main__":
    main()
