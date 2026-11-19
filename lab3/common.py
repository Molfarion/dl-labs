"""Спільний код ЛР3: поділ, дані, перетворення, модель і режими, цикл навчання, журнали."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

LAB_DIR = Path(__file__).resolve().parent
SPLIT_CSV = LAB_DIR / "split.csv"
RUNS_DIR = LAB_DIR / "runs"
FIG_DIR = LAB_DIR / "figures"
RESULTS_DIR = LAB_DIR / "results"
MODEL_PATH = LAB_DIR / "lab3_model.pt"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PARTS = ["train", "validation", "test"]
TRAIN_PER_CLASS = 200
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EPOCHS = 8
TRAIN_BATCH = 32
EVAL_BATCH = 64
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-3
BACKBONE_LR = 3e-4

TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
REGIMES = {
    # trainable: модулі з requires_grad=True; eval: модулі, які set_train_mode переводить у eval()
    "scratch": {"weights": None, "trainable": TOP_MODULES, "eval": []},
    "frozen": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["fc"],
        "eval": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    },
    "unfreeze": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["layer4", "fc"],
        "eval": ["conv1", "bn1", "layer1", "layer2", "layer3"],
    },
}
EXPECTED_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}
EXPECTED_TOTAL = 11_179_590
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

# Контрольна таблиця поділу з умови: (кількість, сума числових імен файлів)
EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1_996_761), "validation": (218, 4_837_323), "test": (219, 4_837_559)},
    "forest": {"train": (200, 2_032_625), "validation": (237, 5_276_962), "test": (237, 5_291_326)},
    "glacier": {"train": (200, 2_183_923), "validation": (276, 6_135_652), "test": (277, 6_157_250)},
    "mountain": {"train": (200, 1_950_345), "validation": (262, 5_789_535), "test": (263, 5_849_765)},
    "sea": {"train": (200, 1_987_613), "validation": (255, 5_655_068), "test": (255, 5_668_011)},
    "street": {"train": (200, 1_841_817), "validation": (250, 5_541_845), "test": (251, 5_547_533)},
}


# ---------------------------------------------------------------- поділ

def part_dir(data_root: Path, part: str) -> Path:
    return data_root / ("seg_train/seg_train" if part == "train" else "seg_test/seg_test")


def build_split(data_root: Path) -> list[dict]:
    """Процедура поділу з умови. Повертає рядки split.csv."""
    rows = []
    rng = np.random.default_rng(0)  # один генератор для всіх класів seg_train
    for c in CLASSES:
        files = sorted(p.name for p in (part_dir(data_root, "train") / c).iterdir())
        idx = np.arange(len(files))
        rng.shuffle(idx)
        for i in sorted(idx[:TRAIN_PER_CLASS], key=lambda i: files[i]):
            rows.append({"file": files[i], "label": c, "part": "train"})
    rng = np.random.default_rng(0)  # новий генератор для seg_test
    for c in CLASSES:
        files = sorted(p.name for p in (part_dir(data_root, "test") / c).iterdir())
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        for part, chosen in (("validation", idx[:cut]), ("test", idx[cut:])):
            for i in sorted(chosen, key=lambda i: files[i]):
                rows.append({"file": files[i], "label": c, "part": part})
    # порядок рядків у файлі: train, validation, test; усередині — клас, потім ім'я файлу
    rows.sort(key=lambda r: (PARTS.index(r["part"]), CLASS_TO_IDX[r["label"]], r["file"]))
    return rows


def write_split(rows: list[dict], path: Path = SPLIT_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: Path = SPLIT_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_control(rows: list[dict]) -> dict:
    """{клас: {частина: (кількість, сума числових імен)}}"""
    table = {c: {p: [0, 0] for p in PARTS} for c in CLASSES}
    for r in rows:
        cell = table[r["label"]][r["part"]]
        cell[0] += 1
        cell[1] += int(Path(r["file"]).stem)
    return {c: {p: tuple(v) for p, v in d.items()} for c, d in table.items()}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- дані

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class SplitDataset(Dataset):
    """Зображення однієї частини поділу, у порядку рядків split.csv."""

    def __init__(self, data_root: Path, part: str, rows: list[dict] | None = None, transform=None):
        rows = read_split() if rows is None else rows
        self.rows = [r for r in rows if r["part"] == part]
        self.paths = [part_dir(data_root, part) / r["label"] / r["file"] for r in self.rows]
        self.targets = [CLASS_TO_IDX[r["label"]] for r in self.rows]
        if transform is None:
            transform = TRAIN_TF if part == "train" else EVAL_TF
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[i]


def train_loader(data_root: Path, rows=None) -> DataLoader:
    # без власного generator/sampler: перемішування з глобального генератора PyTorch
    return DataLoader(SplitDataset(data_root, "train", rows), batch_size=TRAIN_BATCH,
                      shuffle=True, drop_last=False, num_workers=0)


def eval_loader(data_root: Path, part: str, rows=None) -> DataLoader:
    return DataLoader(SplitDataset(data_root, part, rows), batch_size=EVAL_BATCH,
                      shuffle=False, num_workers=0)


# ---------------------------------------------------------------- модель і режими

def build_model(regime: str, seed: int, device: str | torch.device = "cpu"):
    """Кроки 1–5 з умови. Повертає (model, optimizer)."""
    cfg = REGIMES[regime]
    torch.manual_seed(seed)                                                       # 1
    np.random.seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["weights"] else None
    model = resnet18(weights=weights)                                             # 2
    model.fc = nn.Linear(512, len(CLASSES))                                       # 3
    for name, p in model.named_parameters():                                      # 4
        p.requires_grad = name.split(".")[0] in cfg["trainable"]
    model.to(device)                                                              # 5
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": HEAD_LR}]
    elif regime == "frozen":
        groups = [{"params": list(model.fc.parameters()), "lr": HEAD_LR}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": HEAD_LR},
                  {"params": list(model.layer4.parameters()), "lr": BACKBONE_LR}]
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def set_train_mode(model: nn.Module, regime: str) -> None:
    """model.train(), потім eval() для заморожених модулів режиму."""
    model.train()
    for name in REGIMES[regime]["eval"]:
        getattr(model, name).eval()


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(y_true, logits) -> float:
    return float(f1_score(y_true, np.asarray(logits).argmax(1), average="macro",
                          labels=list(range(len(CLASSES)))))


def train_one_epoch(model, optimizer, loader, regime, device, mode_fn=None) -> float:
    (mode_fn or (lambda m: set_train_mode(m, regime)))(model)
    criterion = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


# ---------------------------------------------------------------- журнал

def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "cpu":
        return f"cpu ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"
    return device.type


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=LAB_DIR, capture_output=True, text=True).stdout.strip()
    code_paths = ["lab3/common.py", "lab3/train.py", "lab3/analysis.py", "pyproject.toml", "uv.lock"]
    status = git("status", "--porcelain", "--untracked-files=all", "--", *[f":/{p}" for p in code_paths])
    return {
        "commit": git("rev-parse", "HEAD") or None,
        "uncommitted_changes": bool(status),
        "uncommitted_files": status.splitlines(),
        # SHA-256 файлів коду: однозначно ідентифікує код, навіть якщо зміни ще не закомічені
        "code_sha256": {p: sha256_file(LAB_DIR.parent / p) for p in code_paths
                        if (LAB_DIR.parent / p).exists()},
    }


def run_config(regime: str, seed: int, optimizer: torch.optim.Optimizer) -> dict:
    """Усі налаштування запуску, включно з тими, що лишилися за замовчуванням."""
    groups = []
    for g in optimizer.param_groups:
        groups.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != "params"}
                      | {"n_params": sum(p.numel() for p in g["params"])})
    cfg = REGIMES[regime]
    return {
        "seed": seed,
        "regime": regime,
        "transforms": {
            "train": repr(TRAIN_TF),
            "eval": repr(EVAL_TF),
            "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
            "image_open": 'PIL.Image.open(path).convert("RGB")',
            "test_time_augmentation": False,
        },
        "model": {
            "arch": "torchvision.models.resnet18",
            "weights": f"ResNet18_Weights.{cfg['weights']}" if cfg["weights"] else None,
            "fc": "torch.nn.Linear(512, 6), стандартна ініціалізація",
            "trainable_modules": cfg["trainable"],
            "eval_modules_during_training": cfg["eval"],
            "batchnorm": {"momentum": 0.1, "eps": 1e-5},
            "dtype": "float32",
        },
        "optimizer": {"class": "torch.optim.AdamW", "param_groups": groups},
        "training": {
            "epochs": EPOCHS,
            "steps_per_epoch": -(-1200 // TRAIN_BATCH),
            "total_steps": EPOCHS * -(-1200 // TRAIN_BATCH),
            "train_loader": {"batch_size": TRAIN_BATCH, "shuffle": True, "drop_last": False,
                             "num_workers": 0, "generator": None, "sampler": None},
            "eval_loader": {"batch_size": EVAL_BATCH, "shuffle": False, "num_workers": 0},
            "loss": "torch.nn.CrossEntropyLoss(reduction='mean')",
            "epoch_loss": "sum_b L_b*|b| / 1200 (L_b до кроку оптимізатора)",
            "metric": "sklearn.metrics.f1_score(average='macro') на validation після кожної епохи",
            "lr_schedule": None, "grad_clipping": None, "mixed_precision": False,
            "label_smoothing": 0.0, "class_weights": None, "early_stopping": False,
            "checkpoint_selection": "ваги після 8-ї епохи",
        },
    }


def environment(device: torch.device) -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "device_name": device_name(device),
        "platform": platform.platform(),
    }


def run_dir(regime: str, seed: int) -> Path:
    return RUNS_DIR / f"{regime}_seed{seed}"


def load_run(regime: str, seed: int) -> dict:
    return json.loads((run_dir(regime, seed) / "run.json").read_text(encoding="utf-8"))


def select_run(logs: dict[tuple[str, int], dict]) -> dict:
    """Правило вибору: режим із найбільшим середнім F, у ньому запуск із найбільшим F."""
    final = {k: v["val_macro_f1"][-1] for k, v in logs.items()}
    means = {}
    for (regime, _), f in final.items():
        means.setdefault(regime, []).append(f)
    means = {r: float(np.mean(v)) for r, v in means.items()}
    regime = max(means, key=means.get)
    seed = max((s for r, s in final if r == regime), key=lambda s: final[(regime, s)])
    return {"regime": regime, "seed": seed, "val_macro_f1": final[(regime, seed)], "regime_means": means}
