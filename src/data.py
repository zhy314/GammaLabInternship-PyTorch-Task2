"""读取 CSV、划分训练/验证子集，以及只在训练子集上拟合的预处理。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs


def load_raw(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取训练集与测试集。测试集没有 target。"""
    data_dir = Path(data_dir) if data_dir is not None else configs.DATA_DIR
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def train_val_split(
    frame: pd.DataFrame,
    val_ratio: float = configs.VAL_RATIO,
    seed: int = configs.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按比例打乱后切出验证子集。验证行不参与后续预处理拟合。"""
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio 必须在 (0, 1) 内，当前为 {val_ratio}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    n_val = max(1, int(round(len(frame) * val_ratio)))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    val_df = frame.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def to_array(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """把指定列转成 float 数组，无法解析的单元格记为 NaN。"""
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


class Preprocessor:
    """缺失值用训练子集均值填补，再用训练子集的均值和标准差做标准化。"""

    def __init__(self) -> None:
        self.fill_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Preprocessor":
        x = np.asarray(x, dtype=np.float64)
        fill = np.nanmean(x, axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)
        filled = np.where(np.isnan(x), fill, x)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.fill_ = fill
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64)
        filled = np.where(np.isnan(x), self.fill_, x)
        return (filled - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        self._check_fitted()
        return {
            "fill": self.fill_.tolist(),
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    def load_state_dict(self, state: dict) -> "Preprocessor":
        self.fill_ = np.asarray(state["fill"], dtype=np.float64)
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.std_ = np.asarray(state["std"], dtype=np.float64)
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None or self.fill_ is None:
            raise RuntimeError("Preprocessor 尚未 fit，不能 transform")


def self_check() -> None:
    """确认形状，以及验证集没有参与均值/标准差拟合。"""
    train, test = load_raw()
    assert list(train.columns) == ["id", *configs.FEATURES, configs.TARGET]
    assert list(test.columns) == ["id", *configs.FEATURES]
    train_df, val_df = train_val_split(train)
    x_train = to_array(train_df, configs.FEATURES)
    x_val = to_array(val_df, configs.FEATURES)
    y_train = to_array(train_df, [configs.TARGET]).ravel()
    prep = Preprocessor().fit(x_train)
    z_train = prep.transform(x_train)
    z_val = prep.transform(x_val)
    assert z_train.shape == x_train.shape
    assert z_val.shape == x_val.shape
    assert y_train.shape[0] == x_train.shape[0]
    assert np.allclose(z_train.mean(axis=0), 0.0, atol=1e-8)
    combined = np.vstack([x_train, x_val])
    combined_mean = np.nanmean(combined, axis=0)
    assert not np.allclose(prep.mean_, combined_mean), "验证集泄漏进了预处理统计量"
    restored = Preprocessor().load_state_dict(prep.state_dict())
    assert np.allclose(restored.transform(x_val), z_val)
    print(
        f"数据检查通过：训练子集 {len(train_df)} 行，验证子集 {len(val_df)} 行，"
        f"测试集 {len(test)} 行；标准化只使用训练子集。"
    )


if __name__ == "__main__":
    self_check()
