"""Команда аналізу ЛР3.

Зі збережених журналів: перевірка хешу split.csv, таблиця семи запусків, криві, розклад ефектів, перевірка вибору.
Потім — одноразова тестова оцінка lab3_model.pt на CPU, матриця плутанини, п'ять характерних помилок.
Усі числа друкуються в stdout і записуються в results/analysis.md та results/test_results.json.

    uv run python analysis.py --data <тека з seg_train/ і seg_test/>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torchvision.models import resnet18

from common import (CLASSES, CLASS_TO_IDX, EPOCHS, FIG_DIR, IMG_SIZE, MODEL_PATH, RESULTS_DIR, RUNS, SPLIT_CSV,
                    eval_loader, load_run, macro_f1, part_dir, predict_logits, read_split, run_dir, select_run,
                    sha256_file)

OUT_LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    OUT_LINES.append(s)


def f3(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def fmt_signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.4f}".replace(".", ",")


# ---------------------------------------------------------------- журнали

def load_logs() -> dict:
    current = sha256_file(SPLIT_CSV)
    logs, bad = {}, []
    for regime, seed in RUNS:
        p = run_dir(regime, seed) / "run.json"
        if not p.exists():
            sys.exit(f"ПОМИЛКА: немає журналу {p}")
        log = load_run(regime, seed)
        if log["split_csv_sha256"] != current:
            bad.append(f"{regime}_seed{seed}: {log['split_csv_sha256']}")
        logs[(regime, seed)] = log
    hashes = {v["split_csv_sha256"] for v in logs.values()}
    if bad or len(hashes) != 1:
        sys.exit("ПОМИЛКА: хеш split.csv у журналах не однаковий або не збігається з поточним "
                 f"({current}):\n  " + "\n  ".join(bad))
    out(f"Хеш split.csv однаковий у всіх 7 журналах і збігається з поточним: {current}")
    # узгодженість збережених логітів validation із журналом
    y_val = np.array([CLASS_TO_IDX[r["label"]] for r in read_split() if r["part"] == "validation"])
    for (regime, seed), log in logs.items():
        logits = np.load(run_dir(regime, seed) / "val_logits.npy")
        assert logits.shape == (len(y_val), len(CLASSES)), logits.shape
        assert abs(macro_f1(y_val, logits) - log["val_macro_f1"][-1]) < 1e-9, f"{regime}_seed{seed}"
        assert len(log["train_loss"]) == len(log["val_macro_f1"]) == EPOCHS and log["optimizer_steps"] == 304
    commits = {(v["git"]["commit"], v["git"]["uncommitted_changes"]) for v in logs.values()}
    code = {json.dumps(v["git"]["code_sha256"], sort_keys=True) for v in logs.values()}
    out(f"Коміти в журналах: {sorted(commits)}; різних версій коду (SHA-256 файлів): {len(code)}")
    return logs


def runs_table(logs: dict) -> None:
    out("\n### Сім запусків\n")
    out("| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |")
    out("|---|---|---|---|---|---|")
    for i, ((regime, seed), log) in enumerate(logs.items(), 1):
        out(f"| {i} | {regime} | {seed} | {f3(log['train_loss'][-1])} | {f3(log['val_macro_f1'][-1])} | "
            f"{log['train_time_seconds']:.0f} |")
    dev = {v["environment"]["device_name"] for v in logs.values()}
    out(f"\nПристрій навчання: {', '.join(sorted(dev))}")


def curves(logs: dict) -> None:
    colors = {"scratch": "#444444", "frozen": "#1f77b4", "unfreeze": "#d62728"}
    styles = {1: "-", 2: "--", 3: ":"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ep = np.arange(1, EPOCHS + 1)
    for (regime, seed), log in logs.items():
        kw = dict(color=colors[regime], ls=styles[seed], marker="o", ms=3, label=f"{regime}, зерно {seed}")
        a1.plot(ep, log["train_loss"], **kw)
        a2.plot(ep, log["val_macro_f1"], **kw)
    a1.set(title="Втрата на train", xlabel="епоха", ylabel="CrossEntropy (середнє за епоху)")
    a2.set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.set_xticks(ep)
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "curves.png", dpi=130)
    plt.close(fig)
    out("\nКриві: figures/curves.png")


def effects(logs: dict) -> dict:
    F = {k: v["val_macro_f1"][-1] for k, v in logs.items()}
    fbar = {r: float(np.mean([F[(r, s)] for s in (1, 2, 3)])) for r in ("frozen", "unfreeze")}
    rng = {r: float(max(F[(r, s)] for s in (1, 2, 3)) - min(F[(r, s)] for s in (1, 2, 3)))
           for r in ("frozen", "unfreeze")}
    R = max(rng.values())
    d_pre = fbar["frozen"] - F[("scratch", 1)]
    d_unf = fbar["unfreeze"] - fbar["frozen"]

    def verdict(d):
        return "вартий перевірки (|Δ| > R)" if abs(d) > R else "не відрізняється від розкиду (|Δ| ≤ R)"

    out("\n### Розклад ефектів\n")
    out("| Величина | Значення |")
    out("|---|---|")
    out(f"| F̄_frozen, R_frozen | {f3(fbar['frozen'])}, {f3(rng['frozen'])} |")
    out(f"| F̄_unfreeze, R_unfreeze | {f3(fbar['unfreeze'])}, {f3(rng['unfreeze'])} |")
    out(f"| F_scratch,1 | {f3(F[('scratch', 1)])} |")
    out(f"| R | {f3(R)} |")
    out(f"| Δpre | {fmt_signed(d_pre)} — {verdict(d_pre)} |")
    out(f"| Δunf | {fmt_signed(d_unf)} — {verdict(d_unf)} |")
    out(f"| Δpre + Δunf | {fmt_signed(d_pre + d_unf)} |")
    out(f"\n|Δpre|/R = {abs(d_pre) / R:.1f}, |Δunf|/R = {abs(d_unf) / R:.2f}")
    return {"F": {f"{r}_seed{s}": v for (r, s), v in F.items()}, "Fbar": fbar, "range": rng, "R": R,
            "delta_pre": d_pre, "delta_unf": d_unf, "delta_total": d_pre + d_unf,
            "delta_pre_exceeds_R": abs(d_pre) > R, "delta_unf_exceeds_R": abs(d_unf) > R}


# ---------------------------------------------------------------- вибір і test

def load_selected_model() -> tuple[nn.Module, dict]:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    assert set(ckpt) >= {"state_dict", "regime", "seed", "classes", "img_size"}
    assert ckpt["classes"] == CLASSES and ckpt["img_size"] == IMG_SIZE
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def check_selection(logs: dict, data_root: Path, model: nn.Module, ckpt: dict) -> dict:
    sel = select_run(logs)
    out("\n### Вибір конфігурації (лише за validation)\n")
    out("Середні macro-F1 validation: " + ", ".join(f"{r} {f3(v)}" for r, v in sel["regime_means"].items()))
    out(f"Правило вибору за журналами → {sel['regime']}_seed{sel['seed']} (F = {f3(sel['val_macro_f1'])}); "
        f"lab3_model.pt → {ckpt['regime']}_seed{ckpt['seed']}")
    if (sel["regime"], sel["seed"]) != (ckpt["regime"], ckpt["seed"]):
        sys.exit("ПОМИЛКА: lab3_model.pt не відповідає правилу вибору за журналами")
    # ваги в lab3_model.pt справді належать цьому запуску: логіти validation на CPU ≈ збережені
    saved = np.load(run_dir(ckpt["regime"], ckpt["seed"]) / "val_logits.npy")
    vl = eval_loader(data_root, "validation")
    logits = predict_logits(model, vl, "cpu")
    diff = float(np.abs(logits - saved).max())
    agree = float((logits.argmax(1) == saved.argmax(1)).mean())
    f1 = macro_f1(np.array(vl.dataset.targets), logits)
    out(f"Перевірка ваг: max |логіти val (CPU) − збережені| = {diff:.2e}, збіг argmax {agree:.4f}, "
        f"macro-F1 val на CPU {f3(f1)}")
    if agree < 0.99:
        sys.exit("ПОМИЛКА: ваги lab3_model.pt не відтворюють логіти validation вибраного запуску")
    return sel | {"val_logits_max_abs_diff": diff, "val_argmax_agreement": agree, "val_macro_f1_cpu": f1}


def test_evaluation(model: nn.Module, data_root: Path, sel: dict, eff: dict) -> dict:
    tl = eval_loader(data_root, "test")
    rows = tl.dataset.rows
    y = np.array(tl.dataset.targets)
    logits = predict_logits(model, tl, "cpu")   # єдиний прохід по test
    np.save(RESULTS_DIR / "test_logits.npy", logits.astype(np.float32))
    pred = logits.argmax(1)
    prob = torch.softmax(torch.from_numpy(logits), 1).numpy()
    f1 = macro_f1(y, logits)
    acc = float(accuracy_score(y, pred))
    n = len(y)
    se = math.sqrt(acc * (1 - acc) / n)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))

    out("\n### Тестова оцінка (CPU, один прохід)\n")
    out(f"macro-F1 test = {f3(f1)}; accuracy test p = {f3(acc)} (n = {n}); "
        f"SE = sqrt(p(1−p)/n) = {f3(se)}")
    out(f"macro-F1 validation вибраного запуску = {f3(sel['val_macro_f1'])}; "
        f"test − validation = {fmt_signed(f1 - sel['val_macro_f1'])}")

    out("\nМатриця плутанини (рядки — справжній клас, стовпці — передбачений):\n")
    out("| справжній \\ передбачений | " + " | ".join(CLASSES) + " | усього |")
    out("|---" * (len(CLASSES) + 2) + "|")
    for i, c in enumerate(CLASSES):
        out(f"| **{c}** | " + " | ".join(str(v) for v in cm[i]) + f" | {cm[i].sum()} |")
    np.savetxt(RESULTS_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASSES), comments="")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="передбачений клас", ylabel="справжній клас",
           title=f"Test, {sel['regime']}_seed{sel['seed']}: macro-F1 = {f1:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # п'ять характерних помилок
    pairs = sorted(((cm[i, j], i, j) for i in range(len(CLASSES)) for j in range(len(CLASSES))
                    if i != j and cm[i, j] > 0), key=lambda t: (-t[0], t[1], t[2]))
    files_by_pair = {(i, j): sorted((rows[k]["file"], k) for k in range(n) if y[k] == i and pred[k] == j)
                     for _, i, j in pairs}
    chosen, rank = [], 0
    while len(chosen) < 5 and any(len(v) > rank for v in files_by_pair.values()):
        for cnt, i, j in (pairs if rank else pairs[:5]):
            if len(chosen) < 5 and len(files_by_pair[(i, j)]) > rank:
                chosen.append((cnt, i, j, *files_by_pair[(i, j)][rank]))
        rank += 1
    errors = []
    out("\n### П'ять характерних помилок\n")
    out("| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | Кількість таких помилок у test |")
    out("|---|---|---|---|---|")
    for cnt, i, j, fname, k in chosen:
        errors.append({"file": fname, "true": CLASSES[i], "pred": CLASSES[j], "prob": float(prob[k, j]),
                       "count": int(cnt), "probs": {c: float(prob[k, m]) for m, c in enumerate(CLASSES)}})
        out(f"| {fname} | {CLASSES[i]} | {CLASSES[j]} | {f3(prob[k, j])} | {cnt} |")
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.7))
    for ax, e in zip(np.atleast_1d(axes), errors):
        with Image.open(part_dir(data_root, "test") / e["true"] / e["file"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{e['file']}\nсправжній: {e['true']}\nпередбачено: {e['pred']} ({e['prob']:.2f})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_errors.png", dpi=130)
    plt.close(fig)
    out("\nРисунки: figures/confusion_matrix.png, figures/test_errors.png")
    return {"model": f"{sel['regime']}_seed{sel['seed']}", "test_macro_f1": f1, "test_accuracy": acc, "n": n,
            "accuracy_se": se, "val_macro_f1_selected": sel["val_macro_f1"],
            "test_minus_val": f1 - sel["val_macro_f1"], "confusion_matrix": cm.tolist(),
            "off_diagonal_pairs": [{"true": CLASSES[i], "pred": CLASSES[j], "count": int(c)} for c, i, j in pairs],
            "errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="тека з seg_train/ і seg_test/ набору Intel")
    args = ap.parse_args()
    data_root = args.data.resolve()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)

    logs = load_logs()
    runs_table(logs)
    curves(logs)
    eff = effects(logs)
    model, ckpt = load_selected_model()
    sel = check_selection(logs, data_root, model, ckpt)
    if not eff["delta_unf_exceeds_R"]:
        out("|Δunf| ≤ R: вибір між frozen і unfreeze визначив шум.")
    test = test_evaluation(model, data_root, sel, eff)

    (RESULTS_DIR / "test_results.json").write_text(
        json.dumps({"effects": eff, "selection": sel, "test": test}, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / "analysis.md").write_text("\n".join(OUT_LINES) + "\n", encoding="utf-8")
    print("\nЗаписано: results/analysis.md, results/test_results.json, results/test_logits.npy")


if __name__ == "__main__":
    main()
