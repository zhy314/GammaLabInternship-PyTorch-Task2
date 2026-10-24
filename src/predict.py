"""加载验证集上最好的模型，对测试集推理一次并写出 submission.csv。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from src.data import Preprocessor, load_raw, to_array
from src.model import build_mlp


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_best(summary_path: Path) -> pd.Series:
    summary = pd.read_csv(summary_path, encoding="utf-8")
    if summary.empty:
        raise RuntimeError(f"{summary_path} 为空，请先运行实验")
    ordered = summary.sort_values(["val_rmse", "gap", "exp_name"], ascending=[True, True, True])
    return ordered.iloc[0]


def main() -> None:
    best = select_best(configs.SUMMARY_PATH)
    ckpt_path = configs.ROOT / str(best["ckpt_path"])
    ckpt = load_checkpoint(ckpt_path)
    model = build_mlp(ckpt["in_dim"], ckpt["hidden_sizes"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    prep = Preprocessor().load_state_dict(ckpt["preprocessor"])

    _, test = load_raw()
    features = list(ckpt["features"])
    x_test = prep.transform(to_array(test, features)).astype("float32")
    with torch.no_grad():
        pred = model(torch.tensor(x_test)).numpy()
    if not np.isfinite(pred).all():
        raise RuntimeError("预测中出现非有限值")

    submission = pd.DataFrame({configs.ID_COL: test[configs.ID_COL].astype(str), "prediction": pred})
    if len(submission) != len(test) or not submission[configs.ID_COL].equals(test[configs.ID_COL].astype(str)):
        raise RuntimeError("提交文件的 id 与测试集不一致")
    submission.to_csv(configs.SUBMISSION_PATH, index=False, encoding="utf-8")
    print(
        f"已用 {best['exp_name']}（验证 RMSE {float(best['val_rmse']):.4f}）"
        f"生成 {configs.SUBMISSION_PATH.name}，共 {len(submission)} 行。"
    )


if __name__ == "__main__":
    main()
