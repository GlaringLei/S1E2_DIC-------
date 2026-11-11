"""01 构建 Ren 复现所需的逐秒 AE / DIC 序列（尽量复用 methodScripts 的中间结果）。

时间基准（不做强对齐）
  - AE 与载荷：datasets/data1/AE-载荷.xlsx 已对齐；其“声发射时间”= 标注表 mechanics_time_s（整秒）。
  - DIC：复用 methodScripts/outputs/frame_index.csv（平均 5 fps，t = t0 + frame/5，t0 由载荷互相关估计 = −0.72 s）。
  - 逐秒格：second s ↔ AE 时间 s ↔ DIC 帧 t∈[s, s+1)；阶段按 common.label_times(s)。

复用
  - methodScripts/outputs/frame_features.csv：逐帧相对张拉/滑移（净值）、γmax 统计、面积占比等
  - methodScripts/outputs/band_reference.json：Ren 式参考带与 3 个监测断面
  - methodScripts/outputs/ren_events.csv / ren_tini_sensitivity.csv：局部化起始与加速区间
  - cache/*.npy：只在围岩区重算 U、τ（3 个断面，原 frame_features 只有中段）

输出 outputs/ae_seconds.csv, outputs/dic_seconds.csv, outputs/ae_quiet_periods.json
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MS = HERE.parents[1] / "methodScripts"
sys.path.insert(0, str(MS))
import common as C                                    # noqa: E402

WA = importlib.import_module("03_window_analysis")    # 复用 window_slope / runs
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
AE_XLSX = C.DATA / "data1" / "AE-载荷.xlsx"

SECS = (1, 2, 3)
RATE_W = 5.0               # Ren 速率：5 s 尾窗线性斜率（methodScripts/03 的结论）
E_MOD, NU = 1.0, 0.25      # 只看形状
ENERGY_HALF = 3            # 7×7 网格点 ≈ 原文 15×15 px
NORMS = ("minmax", "p99", "median")


# ------------------------------------------------------------------ AE
def _norm(x, kind):
    if kind == "minmax":
        return (x - x.min()) / (x.max() - x.min())
    if kind == "p99":
        lo, hi = np.percentile(x, [1, 99])
        return (np.clip(x, lo, hi) - lo) / (hi - lo)
    return x / np.median(x[x > 0])


def ae_seconds():
    lab = pd.read_csv(C.LABEL_CSV)
    lab = lab[lab.mechanics_time_s >= 0].copy()
    ra = lab.rise_time_us / lab.amplitude                 # μs/mV = ms/V
    af = lab.ring_count / lab.duration_us * 1e3           # kHz
    for k in NORMS:
        rn, an = _norm(ra, k), _norm(af, k)
        # Ren：归一化后 AF/RA > 1 为拉伸，< 1 为剪切；RA=0 视为拉伸
        lab[f"shear_{k}"] = ((an < rn) & (rn > 0)).astype(int)
    tm, load, _ = C.read_load()
    sec = np.arange(0, int(np.floor(tm[-1])) + 1)
    g = lab.groupby(lab.mechanics_time_s.astype(int))
    df = pd.DataFrame(index=pd.Index(sec, name="sec"))
    df["hits"] = g.size()
    df["ring_sum"] = g.ring_count.sum()
    df["energy_sum"] = g.energy_mV_ms.sum()
    for k in NORMS:
        df[f"shear_hits_{k}"] = g[f"shear_{k}"].sum()
    df = df.fillna(0.0)

    x = pd.read_excel(AE_XLSX)
    x.columns = ["t_load", "load_kN", "t_ae", "rate5"]
    r5 = x[["t_ae", "rate5"]].dropna().set_index("t_ae").rate5
    df["rate5"] = r5.reindex(sec).to_numpy()              # 用户提供：5 s 移动平均撞击率（已对齐）
    roll = lambda s: s.rolling(5, center=True, min_periods=1).sum()
    for k in NORMS:
        sh = roll(df[f"shear_hits_{k}"])
        df[f"shear5_{k}"] = sh / 5.0                              # 5 s MA 剪切撞击率
        df[f"tensile5_{k}"] = (roll(df["hits"]) - sh) / 5.0
        df[f"shear_frac5_{k}"] = sh / roll(df["hits"]).replace(0, np.nan)
    df["cum_hits"] = df.hits.cumsum()
    df["cum_ring"] = df.ring_sum.cumsum()
    df["load_kN"] = np.interp(sec + 0.5, tm, load)
    code, unc = C.label_times(sec.astype(float))
    df["stage_code"], df["uncertain"] = code, unc.astype(int)
    return df.reset_index(), lab


def quiet_periods(ae, t_peak, min_len=5, active=10.0):
    """AE 平静期（峰前）。
    规则 A（主）：逐秒撞击数连续为 0 且 ≥ min_len 秒，前 30 s 平均 ≥ active 次/s（排除加载初期本来就无 AE）。
    规则 B（对照）：5 s MA 撞击率 < 前 30 s 中位数的 20%，持续 ≥ min_len 秒。
    """
    s, h, r5 = ae.sec.to_numpy(), ae.hits.to_numpy(), ae.rate5.to_numpy()
    out = {"A_zero_run": [], "B_rate5_drop": []}
    prev = pd.Series(h).rolling(30, min_periods=10).mean().shift(1).to_numpy()
    for a0, a1 in WA.runs(h == 0, s.astype(float), min_len - 1):
        i = int(a0)
        if a1 < t_peak and prev[i] >= active:
            out["A_zero_run"].append([a0, a1 + 1.0])
    med = pd.Series(r5).rolling(30, min_periods=10).median().shift(1).to_numpy()
    low = (r5 < 0.2 * med) & (med >= active)
    for a0, a1 in WA.runs(low, s.astype(float), min_len - 1):
        if a1 < t_peak:
            out["B_rate5_drop"].append([a0, a1 + 1.0])
    return out


# ------------------------------------------------------------------ DIC
def window_stat(arr, r, c, h):
    blk = np.asarray(arr[:, r - h:r + h + 1, c - h:c + h + 1], dtype=float)
    return np.nanmedian(blk.reshape(blk.shape[0], -1), axis=1)


def energy_shear(ref):
    """围岩区（带外 30 px、7×7 网格点）应变能密度 U 与沿带剪应力 τα（Ren 式 5、6），3 个断面。"""
    a = C.open_cache()
    out = {}
    kk = E_MOD / (1 - NU ** 2)
    for j, s in enumerate(ref["sections"], 1):
        r, c = s["win_E"]
        exx, eyy, exy = (window_stat(a[f], r, c, ENERGY_HALF) for f in ("exx", "eyy", "exy"))
        sx, sy, txy = kk * (exx + NU * eyy), kk * (eyy + NU * exx), E_MOD / (1 + NU) * exy
        sm, sr = (sx + sy) / 2, np.hypot((sx - sy) / 2, txy)
        s1, s2 = sm + sr, sm - sr
        al = np.radians(s["angle_deg"])
        out[f"U_{j}"] = (s1 ** 2 + s2 ** 2 - 2 * NU * s1 * s2) / (2 * E_MOD)
        out[f"tau_{j}"] = 0.5 * (sx - sy) * np.sin(2 * al) + txy * np.cos(2 * al)
    return pd.DataFrame(out)


def dic_seconds():
    f = pd.read_csv(MS / "outputs" / "frame_features.csv")
    ref = json.loads((MS / "outputs" / "band_reference.json").read_text(encoding="utf-8"))
    t = f.t_s.to_numpy()
    for j in SECS:
        f[f"dmag_{j}"] = np.hypot(f[f"dn_net_{j}"], f[f"ds_net_{j}"])
    f["dmag"] = f[[f"dmag_{j}" for j in SECS]].mean(axis=1)
    f["dn_mean"] = f[[f"dn_net_{j}" for j in SECS]].mean(axis=1)
    f["ds_mean"] = f[[f"ds_net_{j}" for j in SECS]].abs().mean(axis=1)
    rate_cols = [*(f"dn_net_{j}" for j in SECS), *(f"ds_net_{j}" for j in SECS),
                 *(f"dmag_{j}" for j in SECS), "dmag", "dn_mean", "ds_mean"]
    half = int(round(RATE_W * C.FPS / 2))
    for c in rate_cols:
        # 回顾式分析（与 Ren 一致）：中心窗斜率 = 尾窗斜率前移 W/2，避免 2.5 s 的相位滞后
        f[f"{c}__rate"] = pd.Series(WA.window_slope(t, f[c].to_numpy(), RATE_W) * 1e3).shift(-half).to_numpy()
    f = pd.concat([f, energy_shear(ref)], axis=1)
    f = f[t >= 0].copy()
    f["sec"] = np.floor(f.t_s).astype(int)
    keep = ["load_kN", "valid_ratio", "crack_ratio", "gmax_p50", "gmax_p95", "gmax_lift", "A_loc", "F_e1_005",
            "eyy_med", "exx_med", "ev2d_med", "U_surr", "tau_band",
            *(f"dn_net_{j}" for j in SECS), *(f"ds_net_{j}" for j in SECS), *(f"dmag_{j}" for j in SECS),
            "dmag", "dn_mean", "ds_mean", *(f"{c}__rate" for c in rate_cols),
            *(f"U_{j}" for j in SECS), *(f"tau_{j}" for j in SECS)]
    d = f.groupby("sec")[keep].mean()
    d["n_frames"] = f.groupby("sec").size()
    return d.reset_index().rename(columns={"load_kN": "load_kN_dic"})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    b = C.stage_boundaries()
    ae, _ = ae_seconds()
    ae.to_csv(OUT / "ae_seconds.csv", index=False, encoding="utf-8-sig")
    q = quiet_periods(ae, b["t_peak"])
    (OUT / "ae_quiet_periods.json").write_text(json.dumps(q, ensure_ascii=False, indent=1), encoding="utf-8")
    dic = dic_seconds()
    dic.to_csv(OUT / "dic_seconds.csv", index=False, encoding="utf-8-sig")
    print("AE 逐秒:", ae.shape, " DIC 逐秒:", dic.shape)
    print("AE 平静期:", q)
    print("剪切撞击占比（全程）:", {k: round(ae[f"shear_hits_{k}"].sum() / ae.hits.sum(), 3) for k in NORMS})


if __name__ == "__main__":
    main()
