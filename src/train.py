"""单次训练：Adam 权重衰减、按验证 RMSE 早停，并保存最佳权重。"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from src.data import Preprocessor, load_raw, to_array, train_val_split
from src.model import build_mlp


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(err ** 2)))


def train_one(cfg: dict) -> dict:
    """训练一个配置。预处理只在训练子集上拟合，返回最佳轮次的误差与曲线。"""
    seed = int(cfg["seed"])
    set_seed(seed)
    hidden_sizes = list(cfg["hidden_sizes"])
    weight_decay = float(cfg["weight_decay"])
    patience = cfg["patience"]
    max_epochs = int(cfg.get("max_epochs", 300))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 1e-2))
    dropout = float(cfg.get("dropout", 0.0))
    val_ratio = float(cfg.get("val_ratio", configs.VAL_RATIO))
    exp_name = str(cfg["exp_name"])

    raw_train, _ = load_raw()
    train_df, val_df = train_val_split(raw_train, val_ratio=val_ratio, seed=seed)
    x_train = to_array(train_df, configs.FEATURES)
    y_train = to_array(train_df, [configs.TARGET]).ravel()
    x_val = to_array(val_df, configs.FEATURES)
    y_val = to_array(val_df, [configs.TARGET]).ravel()

    prep = Preprocessor().fit(x_train)
    z_train = prep.transform(x_train).astype(np.float32)
    z_val = prep.transform(x_val).astype(np.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    x_train_t = torch.tensor(z_train, dtype=torch.float32)
    x_val_t = torch.tensor(z_val, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = build_mlp(x_train_t.shape[1], hidden_sizes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())

    history = {"train_rmse": [], "val_rmse": []}
    best_val = float("inf")
    best_train = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train_t).numpy()
            val_pred = model(x_val_t).numpy()
        train_rmse = rmse(train_pred, y_train)
        val_rmse = rmse(val_pred, y_val)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_train = train_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience is not None and wait >= int(patience):
                break

    if best_state is None:
        raise RuntimeError(f"{exp_name} 没有产生可用权重")
    model.load_state_dict(best_state)

    ckpt_dir = configs.CHECKPOINT_DIR / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"
    metrics = {
        "exp_name": exp_name,
        "n_params": n_params,
        "epochs_ran": len(history["train_rmse"]),
        "best_epoch": best_epoch,
        "train_rmse": best_train,
        "val_rmse": best_val,
        "gap": best_val - best_train,
        "last_train_rmse": history["train_rmse"][-1],
        "last_val_rmse": history["val_rmse"][-1],
        "last_gap": history["val_rmse"][-1] - history["train_rmse"][-1],
    }
    torch.save(
        {
            "model_state": best_state,
            "in_dim": int(x_train_t.shape[1]),
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "preprocessor": prep.state_dict(),
            "features": list(configs.FEATURES),
            "metrics": metrics,
            "history": history,
            "config": {
                "exp_name": exp_name,
                "capacity": cfg.get("capacity"),
                "hidden_sizes": hidden_sizes,
                "weight_decay": weight_decay,
                "patience": patience,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "dropout": dropout,
                "val_ratio": val_ratio,
            },
        },
        ckpt_path,
    )
    metrics["ckpt_path"] = ckpt_path.relative_to(configs.ROOT).as_posix()
    metrics["history"] = history
    return metrics
