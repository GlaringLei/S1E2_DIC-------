"""01 E2 特征提取与预处理：尺度无关、因果的 DIC 特征（借鉴 Dong et al. 2024 SCCM）+ 因果 AE 特征。

为什么要重做（相对 E1）
  E1 的 DIC-Ren 特征有两类问题：
  (1) 全局信息：参考带取"峰值前 5 s"的帧——峰值时刻来自载荷阶段标注，且对此前样本是未来信息；
      U、τ 用全程最大值归一化（未来信息）。
  (2) 尺度：位移、应变、能量等幅值量随单调加载持续增长，后期阶段在特征空间里比早期大 30–50 倍，
      聚类主要在"量级"上分段，特征与时间 |ρ| 中位数 0.87。

E2 的 DIC 特征（DIC-SF，scale-free）
  每帧只用"当前帧"和"5 s 前那一帧"（不用参考带、不用未来帧、不用全程统计量）；
  所有量都是"比例 / 比值 / 分位数之比 / 相关系数"，对应变整体放大 c 倍时保持不变。
  A 累积场形态（Dong SCCM，相对分级）：以本帧 γmax 中位数为单位，统计各相对等级的面积占比
      sf_Plow  γ < 0.5×中位数 的面积占比（几乎未变形区）
      sf_P2 / sf_P4 / sf_P8   γ ≥ 2× / 4× / 8× 中位数 的面积占比（Dong 的高等级占比）
      sf_q98   log2(P98/P50)：面积占比达 2% 的最高相对等级（Dong 的 2% 判据的连续版）
      sf_top5  最高 5% 测点承担的剪应变份额；sf_gini 基尼系数（集中程度）
      sf_e2low |ε2| < 0.5×中位数 的面积占比（Dong ε2 "初始等级"占比 → 压密程度）
      sf_tens  ε1 > |ε2| 的面积占比（拉伸主导区，Dong ε1 场）
      sf_nu    −med(εxx)/med(εyy)（表观泊松比，侧向/轴向）
      sf_band  γ ≥ 4× 中位数 区域中最大连通域的份额（成带程度，不需要参考带）
      sf_crack 失配（裂缝）面积占比
  B 增量场形态（同样的相对分级，作用在"过去 5 s 的应变增量"上：现在哪里在变形）
      inc_P2 / inc_P4 / inc_q98 / inc_top5
      inc_nu   −med(Δεxx)/med(Δεyy)（增量泊松比，>0.5 指示扩容）
      inc_corr 增量场与累积场的空间相关（新变形是否继续集中在老的集中区）
  C 演化动态（无量纲）
      acc_log  log(过去 5 s 应变增量中位数 / 再往前 5 s 的)：变形加速/减速
      q98_slope  sf_q98 在 5 s 内的斜率：集中程度是否在上升

E2 的 AE 特征（AE·因果）
  与 E1 相同的 14 个特征，唯一修改：RA/AF 归一化由"全程 min-max"改为"截至当前所有撞击的中位数"（因果），
  再以 AF/RA = 1 分拉伸/剪切。

预处理与 E1 完全相同（中位数填补 → 鲁棒标准化 → 截尾 ±5 → PCA 90%），直接复用 E1 的函数。
输出 outputs/frame_features_sf.csv, samples.csv, feature_catalog.csv, scaled_*.csv, embed_*.csv, preprocess.json
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT_R = HERE.parent
E1 = ROOT_R / "E1_单模态聚类基线"
REN = ROOT_R / "Ren2026"
sys.path.insert(0, str(REN))
FX = importlib.import_module("02_extract_features")      # Ren2026 特征函数（同时把 methodScripts 加入路径）
import common as C                                       # noqa: E402
MF = importlib.import_module("02_frame_features")        # methodScripts：valid_mask / smooth_nan / gamma_max


def load_by_path(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E1F = load_by_path(E1 / "01_features.py", "e1_features")   # 复用 E1 的 preprocess()
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
W = 5
LAG = int(W * C.FPS)              # 5 s = 25 帧
BAND_LEVEL = 4.0
MIN_BAND_PTS = 10
NU_CLIP = (-1.0, 2.0)

GROUPS = {
    "累积场形态": ["sf_Plow", "sf_P2", "sf_P4", "sf_P8", "sf_q98", "sf_top5", "sf_gini", "sf_e2low",
                   "sf_tens", "sf_nu", "sf_band", "sf_crack"],
    "增量场形态": ["inc_P2", "inc_P4", "inc_q98", "inc_top5", "inc_nu", "inc_corr"],
    "演化动态": ["acc_log", "q98_slope"],
}
MEANING = {
    "sf_Plow": "γ<0.5×中位数 的面积占比（未变形区）",
    "sf_P2": "γ≥2×中位数 的面积占比", "sf_P4": "γ≥4×中位数 的面积占比", "sf_P8": "γ≥8×中位数 的面积占比",
    "sf_q98": "log2(P98/P50)：占比达 2% 的最高相对等级", "sf_top5": "最高 5% 测点的剪应变份额",
    "sf_gini": "剪应变空间分布的基尼系数", "sf_e2low": "|ε2|<0.5×中位数 的面积占比（Dong ε2 初始等级，压密）",
    "sf_tens": "ε1>|ε2| 的面积占比（拉伸主导）", "sf_nu": "表观泊松比 −med(εxx)/med(εyy)",
    "sf_band": "γ≥4×中位数 区域中最大连通域的份额（成带程度）", "sf_crack": "失配（裂缝）面积占比",
    "inc_P2": "增量场 Δγ≥2×中位数 的面积占比", "inc_P4": "增量场 Δγ≥4×中位数 的面积占比",
    "inc_q98": "增量场 log2(P98/P50)", "inc_top5": "增量场最高 5% 份额",
    "inc_nu": "增量泊松比 −med(Δεxx)/med(Δεyy)", "inc_corr": "增量场与累积场的空间相关（log 尺度）",
    "acc_log": "log(本 5 s 增量中位数 / 前 5 s 增量中位数)", "q98_slope": "sf_q98 的 5 s 斜率",
}


# ------------------------------------------------------------------ DIC-SF
def shape_stats(v, prefix):
    """相对分级（SCCM 的尺度无关版）：以中位数为单位。"""
    v = v[np.isfinite(v) & (v > 0)]
    med = np.median(v)
    r = v / med
    srt = np.sort(v)
    n = len(srt)
    out = {f"{prefix}_P2": (r >= 2).mean(), f"{prefix}_P4": (r >= 4).mean(),
           f"{prefix}_q98": np.log2(np.percentile(r, 98)), f"{prefix}_top5": srt[-max(n // 20, 1):].sum() / srt.sum()}
    if prefix == "sf":
        out.update(sf_Plow=(r < 0.5).mean(), sf_P8=(r >= 8).mean(),
                   sf_gini=1 - 2 * np.sum(np.cumsum(srt)) / (n * srt.sum()) + 1 / n)
    return out, med


def frame_fields(a, k):
    sig = np.array(a["sigma"][k])
    ok = MF.valid_mask(sig)
    f = {c: MF.smooth_nan(np.where(ok, np.array(a[c][k], dtype=float), np.nan)) for c in ("exx", "eyy", "exy")}
    ok &= np.isfinite(f["exx"]) & np.isfinite(f["eyy"]) & np.isfinite(f["exy"])
    return f, ok, sig


def principal(f):
    g = MF.gamma_max(f["exx"], f["eyy"], f["exy"])
    m = (f["exx"] + f["eyy"]) / 2
    return g, m + g, m - g


def dic_sf_frames():
    a = C.open_cache()
    valid0 = MF.valid_mask(np.array(a["sigma"][1]))
    interior0 = (MF.INTERIOR & valid0).sum()
    buf = deque(maxlen=LAG + 1)
    rows = []
    for k in range(C.N_FRAMES):
        f, ok, sig = frame_fields(a, k)
        buf.append((f, ok))
        row = {"frame": k, "sf_crack": MF.cracked_mask(sig, valid0).sum() / interior0}
        if k >= 1 and ok.sum() > 1000:
            g, e1, e2 = principal(f)
            st, med = shape_stats(g[ok], "sf")
            row.update(st)
            a2 = np.abs(e2[ok])
            row["sf_e2low"] = (a2 < 0.5 * np.median(a2)).mean()
            row["sf_tens"] = (e1[ok] > a2).mean()
            row["sf_nu"] = float(np.clip(-np.median(f["exx"][ok]) / np.median(f["eyy"][ok]), *NU_CLIP))
            bm = ok & (g >= BAND_LEVEL * med)
            if bm.sum() >= MIN_BAND_PTS:
                lab, n = ndi.label(bm, np.ones((3, 3)))
                row["sf_band"] = ndi.sum(bm, lab, np.arange(1, n + 1)).max() / bm.sum()
            else:
                row["sf_band"] = 0.0
            if k >= LAG + 1:
                f0, ok0 = buf[0]
                both = ok & ok0
                d = {c: f[c] - f0[c] for c in ("exx", "eyy", "exy")}
                dg = MF.gamma_max(d["exx"], d["eyy"], d["exy"])
                st, dmed = shape_stats(dg[both], "inc")
                row.update(st)
                row["inc_med_raw"] = dmed                      # 仅用于构造 acc_log 比值，不作特征
                row["inc_nu"] = float(np.clip(-np.median(d["exx"][both]) / np.median(d["eyy"][both]), *NU_CLIP))
                x, y = np.log(g[both] + 1e-9), np.log(dg[both] + 1e-9)
                row["inc_corr"] = float(np.corrcoef(x, y)[0, 1])
        rows.append(row)
        if k % 250 == 0:
            print(f"\r  DIC 帧 {k}/{C.N_FRAMES}", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df["acc_log"] = np.log(df.inc_med_raw / df.inc_med_raw.shift(LAG))
    return df


def dic_sf_seconds(frames, secs):
    fi = pd.read_csv(C.OUT / "frame_index.csv")[["frame", "t_s"]]
    f = frames.merge(fi, on="frame")
    f = f[f.t_s >= 0].copy()
    f["sec"] = np.floor(f.t_s).astype(int)
    per = f.groupby("sec").mean(numeric_only=True).reindex(secs)
    out = pd.DataFrame({"sec": secs})
    for g, cols in GROUPS.items():
        for c in cols:
            if c == "q98_slope":
                out[c] = FX.past_slope(per["sf_q98"].reset_index(drop=True), W).to_numpy()
            else:
                out[c] = FX.past(per[c].reset_index(drop=True), W).to_numpy()
    return out


# ------------------------------------------------------------------ AE·因果
def ae_causal_seconds():
    """与 E1 相同的 AE 逐秒序列，只把剪切撞击计数换成因果归一化的版本。"""
    ae = pd.read_csv(REN / "outputs" / "ae_seconds.csv")
    lab = pd.read_csv(C.LABEL_CSV)
    lab = lab[lab.mechanics_time_s >= 0].sort_values(["mechanics_time_s", "serial_number"]).reset_index(drop=True)
    ra = lab.rise_time_us / lab.amplitude
    af = lab.ring_count / lab.duration_us * 1e3
    ra_n = ra / ra.expanding().median().replace(0, np.nan)
    af_n = af / af.expanding().median()
    lab["shear"] = ((af_n < ra_n) & (ra > 0)).astype(int)
    sh = lab.groupby(lab.mechanics_time_s.astype(int)).shear.sum()
    ae["shear_hits_minmax"] = sh.reindex(ae.sec).fillna(0).to_numpy()   # FX.ae_features 读取该列名
    return ae, float(lab.shear.mean())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ae, shear_share = ae_causal_seconds()
    FX.CATALOG.clear()
    fa = FX.ae_features(ae, W)
    cat_ae = pd.DataFrame(FX.CATALOG)
    cat_ae["group_zh"] = cat_ae.kind.map({"state": "强度/状态", "mech": "机制", "dynamic": "动态", "trend": "累积"})

    fr_path = OUT / "frame_features_sf.csv"
    if fr_path.exists():
        frames = pd.read_csv(fr_path)
        print("复用已计算的逐帧 DIC-SF 特征")
    else:
        frames = dic_sf_frames()
        frames.to_csv(fr_path, index=False, encoding="utf-8-sig")
    fd = dic_sf_seconds(frames, ae.sec.to_numpy())
    cat_dic = pd.DataFrame([dict(feature=c, modality="DIC", kind=g, ren_section="Dong SCCM（相对分级）",
                                 meaning=MEANING[c], group_zh=g) for g, cols in GROUPS.items() for c in cols])
    cat = pd.concat([cat_ae, cat_dic], ignore_index=True)

    df = ae[["sec", "stage_code", "uncertain", "load_kN"]].merge(fa, on="sec").merge(fd, on="sec", how="left")
    df.to_csv(OUT / "samples.csv", index=False, encoding="utf-8-sig")
    cat.to_csv(OUT / "feature_catalog.csv", index=False, encoding="utf-8-sig")

    sets = {("AE", "全部"): list(cat_ae.feature),
            ("AE", "去趋势"): list(cat_ae[cat_ae.kind != "trend"].feature),
            ("DIC", "全部"): list(cat_dic.feature),
            ("DIC", "累积场"): GROUPS["累积场形态"],
            ("DIC", "增量动态"): GROUPS["增量场形态"] + GROUPS["演化动态"]}
    log = {"W_s": W, "ae_shear_share_causal": shear_share, "n_samples": len(df),
           "n_certain": int(((df.stage_code >= 0) & (df.uncertain == 0)).sum()), "sets": {}}
    for (mod, tag), cols in sets.items():
        Xs, E, pca, info = E1F.preprocess(df[cols])
        emb = pd.DataFrame(E, columns=[f"PC{i + 1}" for i in range(E.shape[1])])
        emb.insert(0, "sec", df.sec)
        emb.to_csv(OUT / f"embed_{mod}_{tag}.csv", index=False, encoding="utf-8-sig")
        if tag == "全部":
            Xs.insert(0, "sec", df.sec)
            Xs.to_csv(OUT / f"scaled_{mod}.csv", index=False, encoding="utf-8-sig")
        load = pd.DataFrame(pca.components_[:2].T, index=[c for c in cols if not info[c].get("dropped")],
                            columns=["PC1", "PC2"])
        log["sets"][f"{mod}_{tag}"] = dict(n_features=len(cols), n_pcs=int(E.shape[1]),
                                           explained=[round(float(v), 4) for v in pca.explained_variance_ratio_],
                                           top_PC1=load.PC1.abs().sort_values(ascending=False).head(5).round(3).to_dict(),
                                           top_PC2=load.PC2.abs().sort_values(ascending=False).head(5).round(3).to_dict(),
                                           features=info)
        print(f"{mod:3s} {tag:5s}: {len(cols):2d} 特征 → {E.shape[1]} 个主成分（累计方差 {pca.explained_variance_ratio_.sum():.2f}）"
              f"；PC1 主要载荷: " + ", ".join(load.PC1.abs().sort_values(ascending=False).head(3).index))
    (OUT / "preprocess.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"AE 因果归一化后剪切撞击占比 {shear_share:.3f}；样本 {log['n_samples']} 秒，评估用 {log['n_certain']} 秒")
    print("缺失率最高的特征：", df[cat.feature].isna().mean().sort_values(ascending=False).head(4).round(3).to_dict())


if __name__ == "__main__":
    main()
