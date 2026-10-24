"""路径、随机种子与实验网格。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
SUBMISSION_PATH = ROOT / "submission.csv"

FEATURES = ["f1", "f2", "f3", "f4"]
TARGET = "target"
ID_COL = "id"

SEED = 42
SEEDS = [42]
VAL_RATIO = 0.2
MAX_EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-2
DROPOUT = 0.0

# 容量由小到大：一层窄网络、两层中等网络、三层宽网络。
CAPACITIES = {
    "small": [8],
    "medium": [32, 16],
    "large": [128, 64, 32],
}
WEIGHT_DECAYS = [0.0, 1e-3, 1e-2]
# None 表示训练满 MAX_EPOCHS；两种设置都只保留验证 RMSE 最低的权重。
PATIENCES = [30, None]
