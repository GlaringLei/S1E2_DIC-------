"""公共配置：路径、DIC 网格、帧时间映射、阶段标注。

所有脚本用 miniconda3/envs/AeDic 的 python 运行，例如
    python -I methodScripts/01_align_cache.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径
HERE = Path(__file__).resolve().parent
S1E2 = HERE.parent
ROOT = S1E2.parent
DATA = ROOT / "datasets"
DIC_DIR = DATA / "HS-2"
LABEL_CSV = DATA / "AE阶段参考标注明细.csv"      # 人工基于载荷判别的阶段（挂在 AE 记录上）
LOAD_CSV = DATA / "载荷原始时间明细.csv"
PEAK_JSON = DATA / "jiazaiSplit" / "HS2_boundaries.json"
CACHE = S1E2 / "cache"                        # 大体量中间数组（已 gitignore）
OUT = HERE / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- DIC 网格
N_FRAMES = 2300
FPS = 5.0
DT = 1.0 / FPS
X0, X1, Y0, Y1, STEP = 935, 1165, 495, 983, 2          # 参考图像像素坐标
NX = (X1 - X0) // STEP + 1                             # 116 列
NY = (Y1 - Y0) // STEP + 1                             # 245 行（行号随像素 y 向下增加）
MM_PER_PX = 0.1881
H_MM = STEP * MM_PER_PX                                # 网格间距 0.376 mm
CELL_AREA = H_MM ** 2                                  # 每测点代表面积 mm²
FIELDS = ("u_c", "v_c", "exx", "eyy", "exy", "sigma")  # u_c,v_c: mm，y 向上为正；exy 为张量剪应变


def frame_path(k: int) -> Path:
    return DIC_DIR / f"HS-2-{k:04d}_0.csv"


def grid_xy_mm():
    """参考网格物理坐标（mm），x 向右，y 向上；与 Vic-2D 的 x_c/y_c 一致。"""
    xs = 0.1881 * (X0 + STEP * np.arange(NX)) - 194.1659
    ys = -0.1881 * (Y0 + STEP * np.arange(NY)) + 145.2481
    return np.meshgrid(xs, ys)          # 形状 (NY, NX)


def open_cache(mode="r"):
    """返回 {field: memmap(N_FRAMES, NY, NX) float32}，缺失点为 NaN。"""
    return {f: np.load(CACHE / f"{f}.npy", mmap_mode=mode) for f in FIELDS}


# ---------------------------------------------------------------- 时间与阶段
def frame_time(k, t0: float):
    """DIC 第 k 帧（文件编号）在力学时钟上的时刻：t = t0 + k/5。t0 由 01 脚本估计。"""
    return t0 + np.asarray(k, dtype=float) * DT


def load_alignment():
    return json.loads((OUT / "alignment.json").read_text(encoding="utf-8"))


STAGE_ORDER = ["pre_initiation", "initiation_to_damage", "damage_to_peak", "post_peak"]
STAGE_ZH = {"pre_initiation": "起裂前", "initiation_to_damage": "起裂至损伤",
            "damage_to_peak": "损伤至峰值", "post_peak": "峰后"}


def stage_boundaries():
    """从人工标注表读取阶段边界（力学时钟，秒）。

    标注挂在 AE 记录上，AE 时间是整秒，所以边界只能确定到 1 s：
    相邻两阶段的边界取后一阶段的最早整秒，±1 s 视为不确定。峰值取载荷最大值时刻及其不确定区间。
    """
    lab = pd.read_csv(LABEL_CSV, usecols=["mechanics_time_s", "stage_reference"])
    lab = lab[lab.stage_reference.isin(STAGE_ORDER)]
    rng = lab.groupby("stage_reference").mechanics_time_s.agg(["min", "max"])
    peak = json.loads(PEAK_JSON.read_text(encoding="utf-8"))
    return {
        "t_ci": float(rng.loc["initiation_to_damage", "min"]),   # 起裂
        "t_cd": float(rng.loc["damage_to_peak", "min"]),         # 损伤
        "t_peak": float(peak["peak_time_s"]),
        "peak_uncertain": [float(v) for v in peak["peak_uncertain_range_s"]],
        "stage_ranges_from_labels": {s: [float(rng.loc[s, "min"]), float(rng.loc[s, "max"])] for s in rng.index},
    }


def label_times(t, b=None, margin=1.0):
    """把任意时刻映射为阶段码 0..3，以及是否落在不确定带（边界 ±margin 或峰值不确定区间）。t<0 记为 -1。"""
    b = b or stage_boundaries()
    t = np.asarray(t, dtype=float)
    code = np.full(t.shape, -1, int)
    code[t >= 0] = 0
    code[t >= b["t_ci"]] = 1
    code[t >= b["t_cd"]] = 2
    code[t >= b["t_peak"]] = 3
    unc = (np.abs(t - b["t_ci"]) <= margin) | (np.abs(t - b["t_cd"]) <= margin)
    unc |= (t >= b["peak_uncertain"][0]) & (t <= b["peak_uncertain"][1])
    return code, unc


def read_load():
    m = pd.read_csv(LOAD_CSV)
    return m.mechanics_time_s.to_numpy(float), m.load_kN.to_numpy(float), m.position_mm.to_numpy(float)
