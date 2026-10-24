"""损失曲线、实验汇总表与对比图。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs


def ensure_output_dirs() -> None:
    configs.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    configs.FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_loss_curve(history: dict, best_epoch: int, title: str, path: Path) -> None:
    plt = _pyplot()
    epochs = range(1, len(history["train_rmse"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_rmse"], label="训练 RMSE")
    ax.plot(epochs, history["val_rmse"], label="验证 RMSE")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"最佳轮次 {best_epoch}")
    ax.set_xlabel("轮次")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def patience_text(patience) -> str:
    if patience is None or (isinstance(patience, str) and patience == "none"):
        return "无早停"
    return f"早停 patience={int(patience)}"


def plot_summary(rows: list[dict], path: Path) -> None:
    """左/右两列对应早停开关，上排是最佳验证 RMSE，下排是末轮过拟合差距。"""
    plt = _pyplot()
    df = pd.DataFrame(rows)
    capacities = [name for name in configs.CAPACITIES if name in set(df["capacity"])]
    weight_decays = [wd for wd in configs.WEIGHT_DECAYS if wd in set(df["weight_decay"])]
    patience_keys = []
    for patience in configs.PATIENCES:
        key = "none" if patience is None else str(int(patience))
        if key in set(df["patience"].astype(str)):
            patience_keys.append((key, patience_text(patience)))

    fig, axes = plt.subplots(2, len(patience_keys), figsize=(6.2 * len(patience_keys), 8), squeeze=False)
    x = range(len(capacities))
    width = 0.8 / max(1, len(weight_decays))
    metric_rows = [("val_rmse", "最佳轮次验证 RMSE"), ("last_gap", "末轮 gap（验证 − 训练）")]

    for col, (pat_key, pat_label) in enumerate(patience_keys):
        subset = df[df["patience"].astype(str) == pat_key]
        for row_i, (metric, ylabel) in enumerate(metric_rows):
            ax = axes[row_i][col]
            for i, wd in enumerate(weight_decays):
                vals = []
                for capacity in capacities:
                    hit = subset[(subset["capacity"] == capacity) & (subset["weight_decay"] == wd)]
                    vals.append(float(hit[metric].mean()) if len(hit) else float("nan"))
                offsets = [pos + (i - (len(weight_decays) - 1) / 2) * width for pos in x]
                ax.bar(offsets, vals, width=width, label=f"wd={wd:g}")
            ax.set_xticks(list(x))
            ax.set_xticklabels(capacities)
            ax.set_ylabel(ylabel)
            ax.set_title(pat_label)
            ax.grid(True, axis="y", alpha=0.3)
            if row_i == 0 and col == 0:
                ax.legend()
    fig.suptitle("容量、权重衰减与早停对比")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(rows: list[dict], path: Path) -> None:
    columns = [
        "exp_name",
        "capacity",
        "hidden_sizes",
        "n_params",
        "weight_decay",
        "patience",
        "seed",
        "epochs_ran",
        "best_epoch",
        "train_rmse",
        "val_rmse",
        "gap",
        "last_train_rmse",
        "last_val_rmse",
        "last_gap",
        "ckpt_path",
    ]
    frame = pd.DataFrame(rows)[columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
