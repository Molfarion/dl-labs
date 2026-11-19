"""Команда навчання ЛР3: поділ, перевірки пунктів 1–2, сім запусків, журнали, вибір і lab3_model.pt.

Test тут не завантажується і не використовується.

    uv run python train.py --data <тека з seg_train/ і seg_test/>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from common import (CLASSES, EPOCHS, EXPECTED_SPLIT, EXPECTED_TOTAL, EXPECTED_TRAINABLE, FIG_DIR, IMG_SIZE,
                    MEAN, MODEL_PATH, PARTS, REGIMES, RESULTS_DIR, RUNS, RUNS_DIR, SPLIT_CSV, STD, TOP_MODULES,
                    build_model, build_split, environment, eval_loader, git_info, load_run, macro_f1,
                    part_dir, pick_device, predict_logits, read_split, run_config, run_dir, select_run,
                    set_train_mode, sha256_file, split_control, train_loader, train_one_epoch, write_split)

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def yn(b):
    return "так" if b else "ні"


# ---------------------------------------------------------------- 1. перевірки до навчання

def check_split(data_root: Path) -> dict:
    rows = build_split(data_root)
    write_split(rows)
    rows = read_split()  # далі все — лише з split.csv
    control = split_control(rows)
    cells, all_ok = {}, True
    for c in CLASSES:
        for p in PARTS:
            ok = control[c][p] == EXPECTED_SPLIT[c][p]
            all_ok &= ok
            cells[f"{c}/{p}"] = {"got": control[c][p], "expected": EXPECTED_SPLIT[c][p], "match": ok}
    files = {p: {r["file"] for r in rows if r["part"] == p} for p in PARTS}
    overlaps = {f"{a}&{b}": len(files[a] & files[b])
                for i, a in enumerate(PARTS) for b in PARTS[i + 1:]}
    sizes = {}
    for p in ("train", "test"):
        for c in CLASSES:
            for f in (part_dir(data_root, p) / c).iterdir():
                with Image.open(f) as im:
                    sizes[im.size] = sizes.get(im.size, 0) + 1
    print("\n== Поділ: кількість / сума числових імен ==")
    print("| Клас | train | validation | test |")
    for c in CLASSES:
        print(f"| {c} | " + " | ".join(
            f"{control[c][p][0]} / {control[c][p][1]:,}".replace(",", " ") +
            ("" if control[c][p] == EXPECTED_SPLIT[c][p] else " (!)") for p in PARTS) + " |")
    tot = {p: (sum(control[c][p][0] for c in CLASSES), sum(control[c][p][1] for c in CLASSES)) for p in PARTS}
    print("| Разом | " + " | ".join(f"{n} / {s:,}".replace(",", " ") for n, s in tot.values()) + " |")
    print(f"Усі 18 клітинок збігаються з контрольною таблицею: {yn(all_ok)}; перетини частин: {overlaps}")
    print(f"Розміри зображень seg_train+seg_test: {len(sizes)} різних; не 150x150: "
          f"{sum(n for s, n in sizes.items() if s != (150, 150))}")
    assert all_ok, "поділ не збігається з контрольною таблицею"
    assert all(v == 0 for v in overlaps.values()), "файл входить у дві частини"
    return {"cells": cells, "totals": tot, "all_match": all_ok, "overlaps": overlaps,
            "split_sha256": sha256_file(SPLIT_CSV), "n_images": sum(sizes.values()),
            "n_not_150x150": sum(n for s, n in sizes.items() if s != (150, 150)),
            "sizes": {f"{w}x{h}": n for (w, h), n in sorted(sizes.items(), key=lambda kv: -kv[1])}}


def save_batch_figure(data_root: Path) -> str:
    torch.manual_seed(0)
    x, y = next(iter(train_loader(data_root)))
    img = (x[:16] * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)).clamp(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.8))
    for ax, im, lab in zip(axes.flat, img, y[:16]):
        ax.imshow(im.permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(lab)], fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Навчальний пакет після перетворень (перші 16 із {len(y)}, {IMG_SIZE}x{IMG_SIZE}, "
                 f"обернене нормування)", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "train_batch.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out.relative_to(out.parents[1]))


def check_initial_loss(data_root: Path, device) -> dict:
    res = {}
    for regime in ("scratch", "frozen"):
        model, _ = build_model(regime, 1, device)
        x, y = next(iter(train_loader(data_root)))
        set_train_mode(model, regime)
        with torch.no_grad():
            logits = model(x.to(device))
            loss = nn.functional.cross_entropy(logits, y.to(device)).item()
        p = logits.softmax(1)
        res[regime] = {"L0": loss, "ln6": math.log(6), "diff": loss - math.log(6),
                       "logit_std": logits.std().item(), "logit_abs_max": logits.abs().max().item(),
                       "mean_max_prob": p.max(1).values.mean().item()}
    print("\n== Початкова втрата L0 (зерно 1, перший пакет, set_train_mode, no_grad) ==")
    for r, v in res.items():
        print(f"{r:8s} L0={v['L0']:.4f}  ln6={v['ln6']:.4f}  L0-ln6={v['diff']:+.4f}  "
              f"std логітів={v['logit_std']:.3f}  max|логіт|={v['logit_abs_max']:.3f}  "
              f"середня max p={v['mean_max_prob']:.3f}")
    return res


# ---------------------------------------------------------------- 2. перевірка режимів

def check_params(device) -> dict:
    res = {}
    print("\n== Параметри та групи оптимізатора (сума numel) ==")
    for regime in REGIMES:
        model, opt = build_model(regime, 1, device)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        groups = [{"n_params": sum(p.numel() for p in g["params"]), "lr": g["lr"],
                   "weight_decay": g["weight_decay"]} for g in opt.param_groups]
        in_opt = [p for g in opt.param_groups for p in g["params"]]
        frozen_in_opt = sum(1 for p in in_opt if not p.requires_grad)
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        all_trainable_in_opt = trainable_ids == {id(p) for p in in_opt} and len(in_opt) == len(trainable_ids)
        ok = (total == EXPECTED_TOTAL and trainable == EXPECTED_TRAINABLE[regime] and frozen_in_opt == 0
              and all_trainable_in_opt and sum(g["n_params"] for g in groups) == trainable)
        res[regime] = {"total": total, "trainable": trainable, "groups": groups,
                       "requires_grad_false_in_optimizer": frozen_in_opt,
                       "all_trainable_in_optimizer": all_trainable_in_opt, "match": ok}
        gtxt = "; ".join(f"{g['n_params']:,} @ lr={g['lr']:g}".replace(",", " ") for g in groups)
        print(f"{regime:8s} усього={total:,} навчаються={trainable:,} групи: {gtxt}; "
              f"requires_grad=False в оптимізаторі: {frozen_in_opt}; збіг: {yn(ok)}".replace(",", " "))
        assert ok, f"параметри режиму {regime} не збігаються з таблицею"
    return res


def changes_after_steps(regime: str, data_root: Path, device, mode_fn=None, steps: int = 5):
    model, opt = build_model(regime, 1, device)
    before = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
    param_names = {n for n, _ in model.named_parameters()}
    loader = train_loader(data_root)
    (mode_fn or (lambda m: set_train_mode(m, regime)))(model)
    criterion = nn.CrossEntropyLoss()
    it = iter(loader)
    for _ in range(steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()
    after = model.state_dict()
    rows = []
    for mod in TOP_MODULES:
        keys = [k for k in after if k.split(".")[0] == mod]
        pk = [k for k in keys if k in param_names]
        bk = [k for k in keys if k.endswith(BN_BUFFERS)]
        p_changed = any(not torch.equal(before[k], after[k]) for k in pk)
        b_changed = any(not torch.equal(before[k], after[k]) for k in bk) if bk else None
        frozen = mod in REGIMES[regime]["eval"]
        exp_p = not frozen
        exp_b = None if not bk else (not frozen)
        rows.append({"module": mod, "params_changed": p_changed, "bn_changed": b_changed,
                     "expected_params_changed": exp_p, "expected_bn_changed": exp_b,
                     "match": p_changed == exp_p and b_changed == exp_b})
    detail = {}
    for buf in ("running_mean", "running_var", "num_batches_tracked"):
        k = f"layer1.0.bn1.{buf}"
        detail[k] = float((after[k].double() - before[k].double()).abs().max())
    return rows, detail


def print_changes(title, regime, rows):
    print(f"\n== {title} ==")
    print("| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |")
    for r in rows:
        bn = "немає BN" if r["bn_changed"] is None else yn(r["bn_changed"])
        print(f"| {regime} | {r['module']} | {yn(r['params_changed'])} | {bn} | {yn(r['match'])} |")


def check_changes(data_root: Path, device) -> dict:
    res = {}
    for regime in ("frozen", "unfreeze"):
        rows, detail = changes_after_steps(regime, data_root, device)
        print_changes(f"Що змінюється за 5 кроків, {regime}, set_train_mode", regime, rows)
        res[regime] = {"rows": rows, "layer1.0.bn1_max_abs_change": detail}
        assert all(r["match"] for r in rows), f"режим {regime}: зміни не збігаються з очікуваними"
    return res


def bn_bug_experiment(data_root: Path, device) -> dict:
    rows, detail = changes_after_steps("frozen", data_root, device, mode_fn=lambda m: m.train())
    print_changes("Дослід із помилкою: frozen, model.train() замість set_train_mode", "frozen", rows)
    for k, v in detail.items():
        print(f"max |Δ {k}| = {v:.6g}")
    return {"rows": rows, "layer1.0.bn1_max_abs_change": detail}


# ---------------------------------------------------------------- 3. сім запусків

def train_run(regime: str, seed: int, data_root: Path, device) -> dict:
    model, opt = build_model(regime, seed, device)
    cfg = run_config(regime, seed, opt)
    tl = train_loader(data_root)
    vl = eval_loader(data_root, "validation")
    y_val = np.array(vl.dataset.targets)
    assert len(tl) == 38 and len(tl.dataset) % 32 == 16
    losses, f1s, epoch_times, steps = [], [], [], 0
    t_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, opt, tl, regime, device)
        steps += len(tl)
        logits = predict_logits(model, vl, device)
        f1 = macro_f1(y_val, logits)
        losses.append(loss)
        f1s.append(f1)
        epoch_times.append(time.perf_counter() - t0)
        print(f"  {regime}_seed{seed} епоха {epoch}: втрата train {loss:.4f}, macro-F1 val {f1:.4f} "
              f"({epoch_times[-1]:.0f} с)", flush=True)
    train_time = time.perf_counter() - t_start
    assert steps == 304
    out = run_dir(regime, seed)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    log = {
        **cfg,
        "git": git_info(),
        "split_csv_sha256": sha256_file(SPLIT_CSV),
        "environment": environment(device),
        "train_loss": losses,
        "val_macro_f1": f1s,
        "optimizer_steps": steps,
        "epoch_seconds": epoch_times,
        "train_time_seconds": train_time,
        "train_time_note": "навчання й оцінка validation після кожної епохи, 8 епох",
        "val_logits": {"file": "val_logits.npy", "shape": list(logits.shape)},
    }
    (out / "run.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="тека з seg_train/ і seg_test/ набору Intel")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--skip-existing", action="store_true",
                    help="не перенавчати запуски, для яких уже є run.json і model.pt на поточному split.csv")
    args = ap.parse_args()
    data_root = args.data.resolve()
    device = pick_device(args.device)
    for d in (FIG_DIR, RESULTS_DIR, MODEL_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Пристрій: {device} ({environment(device)['device_name']}), дані: {data_root}")

    checks = {"split": check_split(data_root)}
    checks["batch_figure"] = save_batch_figure(data_root)
    checks["initial_loss"] = check_initial_loss(data_root, device)
    checks["params"] = check_params(device)
    checks["changes"] = check_changes(data_root, device)
    checks["bn_bug"] = bn_bug_experiment(data_root, device)
    checks["environment"] = environment(device)
    (RESULTS_DIR / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nУсі перевірки пунктів 1–2 збіглися з очікуваними; переходжу до семи запусків.")

    split_hash = sha256_file(SPLIT_CSV)
    logs = {}
    for regime, seed in RUNS:
        d = run_dir(regime, seed)
        if (args.skip_existing and (d / "run.json").exists() and (d / "model.pt").exists()
                and load_run(regime, seed)["split_csv_sha256"] == split_hash):
            print(f"\n[{regime}_seed{seed}] уже є, пропускаю")
            logs[(regime, seed)] = load_run(regime, seed)
            continue
        print(f"\n[{regime}_seed{seed}]", flush=True)
        logs[(regime, seed)] = train_run(regime, seed, data_root, device)
        print(f"  час навчання {logs[(regime, seed)]['train_time_seconds']:.0f} с")

    # 5. вибір — лише за validation
    sel = select_run(logs)
    state = torch.load(run_dir(sel["regime"], sel["seed"]) / "model.pt", map_location="cpu")
    torch.save({"state_dict": state, "regime": sel["regime"], "seed": sel["seed"],
                "classes": CLASSES, "img_size": IMG_SIZE}, MODEL_PATH)
    (RUNS_DIR / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nСередні macro-F1 validation: {sel['regime_means']}")
    print(f"Вибрано {sel['regime']}_seed{sel['seed']} (macro-F1 val {sel['val_macro_f1']:.4f}) → {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
