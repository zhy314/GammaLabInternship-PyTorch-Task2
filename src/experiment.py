"""遍历容量、权重衰减和早停网格，写出汇总表与曲线。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from src.data import load_raw, to_array, train_val_split
from src.train import rmse, train_one
from src.utils import ensure_output_dirs, plot_loss_curve, plot_summary, write_summary


def make_exp_name(capacity: str, weight_decay: float, patience, seed: int) -> str:
    if weight_decay == 0:
        wd_txt = "0"
    else:
        wd_txt = f"{weight_decay:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    pat_txt = "none" if patience is None else str(int(patience))
    return f"{capacity}_wd{wd_txt}_pat{pat_txt}_seed{seed}"


def build_grid() -> list[dict]:
    grid = []
    for capacity, hidden_sizes in configs.CAPACITIES.items():
        for weight_decay in configs.WEIGHT_DECAYS:
            for patience in configs.PATIENCES:
                for seed in configs.SEEDS:
                    grid.append(
                        {
                            "exp_name": make_exp_name(capacity, weight_decay, patience, seed),
                            "capacity": capacity,
                            "hidden_sizes": list(hidden_sizes),
                            "weight_decay": weight_decay,
                            "patience": patience,
                            "seed": seed,
                            "max_epochs": configs.MAX_EPOCHS,
                            "batch_size": configs.BATCH_SIZE,
                            "lr": configs.LR,
                            "dropout": configs.DROPOUT,
                            "val_ratio": configs.VAL_RATIO,
                        }
                    )
    return grid


def mean_baseline_rmse(seed: int) -> float:
    """用训练子集目标均值预测验证子集，作为不看特征的参照。"""
    raw_train, _ = load_raw()
    train_df, val_df = train_val_split(raw_train, val_ratio=configs.VAL_RATIO, seed=seed)
    y_train = to_array(train_df, [configs.TARGET]).ravel()
    y_val = to_array(val_df, [configs.TARGET]).ravel()
    pred = np.full_like(y_val, float(y_train.mean()))
    return rmse(pred, y_val)


def main() -> None:
    ensure_output_dirs()
    grid = build_grid()
    rows = []
    print(f"共 {len(grid)} 组实验，验证集比例 {configs.VAL_RATIO}，最多 {configs.MAX_EPOCHS} 轮。")
    for seed in configs.SEEDS:
        print(f"种子 {seed} 的训练均值基线验证 RMSE：{mean_baseline_rmse(seed):.4f}")
    for index, cfg in enumerate(grid, start=1):
        metrics = train_one(cfg)
        history = metrics.pop("history")
        figure_path = configs.FIGURE_DIR / f"{cfg['exp_name']}_loss_curve.png"
        plot_loss_curve(history, metrics["best_epoch"], cfg["exp_name"], figure_path)
        row = {
            "exp_name": metrics["exp_name"],
            "capacity": cfg["capacity"],
            "hidden_sizes": "-".join(str(size) for size in cfg["hidden_sizes"]),
            "n_params": metrics["n_params"],
            "weight_decay": cfg["weight_decay"],
            "patience": "none" if cfg["patience"] is None else str(int(cfg["patience"])),
            "seed": cfg["seed"],
            "epochs_ran": metrics["epochs_ran"],
            "best_epoch": metrics["best_epoch"],
            "train_rmse": round(metrics["train_rmse"], 6),
            "val_rmse": round(metrics["val_rmse"], 6),
            "gap": round(metrics["gap"], 6),
            "last_train_rmse": round(metrics["last_train_rmse"], 6),
            "last_val_rmse": round(metrics["last_val_rmse"], 6),
            "last_gap": round(metrics["last_gap"], 6),
            "ckpt_path": metrics["ckpt_path"],
        }
        rows.append(row)
        print(
            f"[{index}/{len(grid)}] {row['exp_name']}  "
            f"best val={row['val_rmse']:.4f}  gap={row['gap']:.4f}  "
            f"last_gap={row['last_gap']:.4f}  epochs={row['epochs_ran']}"
        )
    write_summary(rows, configs.SUMMARY_PATH)
    plot_summary(rows, configs.FIGURE_DIR / "summary_compare.png")
    best = min(rows, key=lambda item: (item["val_rmse"], item["gap"], item["exp_name"]))
    print(f"汇总已写入 {configs.SUMMARY_PATH.relative_to(configs.ROOT).as_posix()}")
    print(f"验证 RMSE 最低：{best['exp_name']}  val={best['val_rmse']:.4f}")


if __name__ == "__main__":
    main()
