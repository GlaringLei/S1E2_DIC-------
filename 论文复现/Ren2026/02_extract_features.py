"""02 按 Ren 方法（00_Ren方法伪代码.md §9）提取逐秒 AE / DIC 特征。

每 1 s 一个样本，特征只用过去 W 秒（W = 5 主口径，10 作敏感性）的数据，便于后续在线识别。
复用：01 产出的 ae_seconds.csv / dic_seconds.csv（DIC 逐帧量已按秒平均）、AE 逐撞击记录（波形参数中位数）。

每个特征带一个 kind 标签：
  trend   累积量，随加载单调增长，在单试样上近似"时钟"
  state   当前状态（强度、局部化程度等）
  dynamic 变化率 / 波动 / 平静程度
  mech    破裂机制（拉伸/剪切、RA、AF）
输出 outputs/features_W{W}.csv（sec, stage_code, uncertain, load_kN, 各特征）
     outputs/feature_catalog.csv（特征名、模态、kind、Ren 出处、含义）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MS = HERE.parents[1] / "methodScripts"
sys.path.insert(0, str(MS))
import common as C                                    # noqa: E402

OUT = HERE / "outputs"
NORM = "minmax"
SECS = (1, 2, 3)
WINDOWS = (5, 10)

CATALOG = []


def reg(name, modality, kind, src, desc):
    CATALOG.append(dict(feature=name, modality=modality, kind=kind, ren_section=src, meaning=desc))


def past(s: pd.Series, W, how="mean"):
    r = s.rolling(W, min_periods=max(2, W // 2))
    return getattr(r, how)()


def past_slope(s: pd.Series, W):
    """过去 W 秒的线性斜率（每秒）。"""
    x = np.arange(W, dtype=float) - (W - 1) / 2
    return s.rolling(W, min_periods=W).apply(lambda v: np.dot(x, v - v.mean()) / np.dot(x, x), raw=True)


def ae_waveform_medians(W):
    """RA、AF、幅度、持续时间：过去 W 秒内所有撞击的中位数（逐撞击记录计算，避免逐秒中位数再平均）。"""
    lab = pd.read_csv(C.LABEL_CSV, usecols=["mechanics_time_s", "amplitude", "rise_time_us",
                                             "duration_us", "ring_count"])
    lab = lab[lab.mechanics_time_s >= 0]
    lab = lab.assign(sec=lab.mechanics_time_s.astype(int),
                     RA=lab.rise_time_us / lab.amplitude,
                     AF=lab.ring_count / lab.duration_us * 1e3)
    tm, _, _ = C.read_load()
    secs = np.arange(0, int(np.floor(tm[-1])) + 1)
    out = {c: np.full(len(secs), np.nan) for c in ("RA", "AF", "amplitude", "duration_us")}
    by = {s: g for s, g in lab.groupby("sec")}
    for i, s in enumerate(secs):
        parts = [by[k] for k in range(s - W + 1, s + 1) if k in by]
        if parts:
            g = pd.concat(parts)
            for c in out:
                out[c][i] = g[c].median()
    return pd.DataFrame({"sec": secs, **out})


def ae_features(ae, W):
    f = pd.DataFrame({"sec": ae.sec})
    hits = ae.hits
    sh = ae[f"shear_hits_{NORM}"]
    tot = past(hits, W, "sum")
    f["ae_rate"] = np.log1p(past(hits, W))
    reg("ae_rate", "AE", "state", "§6", "log(1+过去 W 秒平均撞击率)")
    f["ae_ring_rate"] = np.log1p(past(ae.ring_sum, W))
    reg("ae_ring_rate", "AE", "state", "§6", "log(1+平均振铃计数率)")
    f["ae_energy_rate"] = np.log1p(past(ae.energy_sum, W))
    reg("ae_energy_rate", "AE", "state", "§6", "log(1+平均能量率)")
    f["ae_shear_rate"] = np.log1p(past(sh, W))
    reg("ae_shear_rate", "AE", "mech", "§6", "log(1+剪切型撞击率)")
    f["ae_tensile_rate"] = np.log1p(past(hits - sh, W))
    reg("ae_tensile_rate", "AE", "mech", "§6", "log(1+拉伸型撞击率)")
    f["ae_shear_frac"] = past(sh, W, "sum") / tot.where(tot >= 5)
    reg("ae_shear_frac", "AE", "mech", "§6", "剪切型占比（窗口内撞击 ≥5 才计算）")
    wm = ae_waveform_medians(W)
    for c, nm in (("RA", "ae_RA_med"), ("AF", "ae_AF_med"), ("amplitude", "ae_amp_med"), ("duration_us", "ae_dur_med")):
        f[nm] = np.log(wm[c].to_numpy() + 1e-3) if c in ("RA", "AF") else wm[c].to_numpy()
        reg(nm, "AE", "mech", "§6", f"过去 W 秒撞击 {c} 中位数" + ("（取对数）" if c in ("RA", "AF") else ""))
    f["ae_zero_frac"] = past((hits == 0).astype(float), W)
    reg("ae_zero_frac", "AE", "dynamic", "§7", "零撞击秒占比（平静程度）")
    f["ae_rate_slope"] = past_slope(np.log1p(hits), W)
    reg("ae_rate_slope", "AE", "dynamic", "§7", "log 撞击率的斜率")
    f["ae_rate_cv"] = past(hits, W, "std") / past(hits, W).replace(0, np.nan)
    reg("ae_rate_cv", "AE", "dynamic", "§7", "撞击率变异系数（突发性）")
    f["ae_cum_hits"] = np.log1p(ae.cum_hits)
    reg("ae_cum_hits", "AE", "trend", "§7", "log(1+累计撞击数)")
    return f


def dic_features(dic, W):
    f = pd.DataFrame({"sec": dic.sec})
    dn = dic[[f"dn_net_{j}" for j in SECS]].mean(axis=1) * 1e3
    ds = dic[[f"ds_net_{j}" for j in SECS]].abs().mean(axis=1) * 1e3
    dm = dic.dmag * 1e3
    f["dic_dn"], f["dic_ds"], f["dic_dmag"] = past(dn, W), past(ds, W), past(dm, W)
    for nm, d in (("dic_dn", "相对张拉 δn（μm，三断面平均）"), ("dic_ds", "相对滑移 |δs|（μm）"),
                  ("dic_dmag", "合位移 |δ|（μm）")):
        reg(nm, "DIC", "trend", "§3", d)
    # 速率：01 中的 5 s 中心窗斜率会用到未来 2.5 s；这里改为对逐秒值做过去 W 秒斜率（因果）
    f["dic_dn_rate"] = past_slope(dn, W)
    f["dic_ds_rate"] = past_slope(ds, W)
    f["dic_dmag_rate"] = past_slope(dm, W)
    for nm, d in (("dic_dn_rate", "δn 速率 μm/s"), ("dic_ds_rate", "|δs| 速率 μm/s"), ("dic_dmag_rate", "|δ| 速率 μm/s")):
        reg(nm, "DIC", "dynamic", "§4", d)
    base = f.dic_dmag_rate.rolling(30, min_periods=10).median().shift(W)
    f["dic_acc_ratio"] = np.log((f.dic_dmag_rate.clip(lower=0) + 0.05) / (base.clip(lower=0) + 0.05))
    reg("dic_acc_ratio", "DIC", "dynamic", "§4", "log(当前 |δ̇| / 此前 30 s 中位速率)，加速演化指示")
    per = pd.concat([past_slope(dic[f"dmag_{j}"] * 1e3, W) for j in SECS], axis=1)
    f["dic_section_cv"] = per.std(axis=1) / per.mean(axis=1).abs().clip(lower=0.2)
    reg("dic_section_cv", "DIC", "dynamic", "§4", "三断面速率变异系数（不同步程度）")
    U = dic[[f"U_{j}" for j in SECS]].mean(axis=1)
    tau = dic[[f"tau_{j}" for j in SECS]].mean(axis=1)
    f["dic_U"] = past(U, W) / U.max()
    f["dic_tau"] = past(tau, W) / tau.abs().max()
    reg("dic_U", "DIC", "state", "§5", "围岩应变能密度（E=1，按全程最大值归一）")
    reg("dic_tau", "DIC", "state", "§5", "带上剪应力（同上）")
    f["dic_U_slope"] = past_slope(U / U.max(), W)
    f["dic_tau_slope"] = past_slope(tau / tau.abs().max(), W)
    reg("dic_U_slope", "DIC", "dynamic", "§5", "U 变化率（能量积累/释放）")
    reg("dic_tau_slope", "DIC", "dynamic", "§5", "τα 变化率（应力转移/释放）")
    f["dic_gmax_p50"] = past(dic.gmax_p50, W) * 1e3
    f["dic_gmax_lift"] = past(dic.gmax_lift, W) * 1e3
    reg("dic_gmax_p50", "DIC", "state", "§1", "γmax 中位数 ‰（整体剪切水平）")
    reg("dic_gmax_lift", "DIC", "state", "§1", "γmax P95−P50 ‰（局部化强度）")
    f["dic_gmax_lift_slope"] = past_slope(dic.gmax_lift * 1e3, W)
    reg("dic_gmax_lift_slope", "DIC", "dynamic", "§1", "局部化强度变化率")
    f["dic_eyy"] = -past(dic.eyy_med, W) * 1e3
    reg("dic_eyy", "DIC", "trend", "补充", "轴向压应变中位数 ‰（整体变形）")
    f["dic_crack"] = past(dic.crack_ratio, W) * 100
    reg("dic_crack", "DIC", "state", "补充", "失配（裂缝）面积占比 %")
    return f


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ae = pd.read_csv(OUT / "ae_seconds.csv")
    dic = pd.read_csv(OUT / "dic_seconds.csv")
    dic = dic[dic.sec.isin(ae.sec)].reset_index(drop=True)
    for W in WINDOWS:
        CATALOG.clear()
        fa = ae_features(ae, W)
        fd = dic_features(dic, W)
        df = ae[["sec", "stage_code", "uncertain", "load_kN"]].merge(fa, on="sec").merge(fd, on="sec", how="left")
        df.to_csv(OUT / f"features_W{W}.csv", index=False, encoding="utf-8-sig")
        if W == WINDOWS[0]:
            cat = pd.DataFrame(CATALOG)
            cat.to_csv(OUT / "feature_catalog.csv", index=False, encoding="utf-8-sig")
        print(f"W={W}s: {df.shape[0]} 秒 × {df.shape[1] - 4} 特征；含缺失的行 {int(df.iloc[:, 4:].isna().any(axis=1).sum())}")
    print(cat.groupby(["modality", "kind"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
