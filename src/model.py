"""可配置隐藏层宽度与深度的回归 MLP。"""
from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """多层感知机：每层线性变换后接 ReLU，最后一层输出标量。"""

    def __init__(self, in_dim: int, hidden_sizes: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim 必须为正")
        if not hidden_sizes:
            raise ValueError("hidden_sizes 不能为空")
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden_sizes:
            if width <= 0:
                raise ValueError(f"隐藏层宽度必须为正，当前为 {width}")
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_mlp(in_dim: int, hidden_sizes: list[int], dropout: float = 0.0) -> MLP:
    return MLP(in_dim, list(hidden_sizes), dropout)
