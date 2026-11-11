"""01 E3 特征：Dong SCCM 原文（固定量程）应变集中特征 + 复用 E2 的尺度无关特征。

Dong et al. (2024) SCCM 原文做法（数值化，不经图像颜色）：
  ε1 场按固定量程 [0, 0.03] 均分 32 级，ε2 场按 [−0.01, 0] 均分 32 级（原文常用 32），
  每帧统计每一级的面积占比；
  判据 1（微损伤临界应变 εmd）：以"之前一直 < 2%、之后突增"的最低等级为 εmd；
  判据 2（压密结束）：ε2 最靠近 0 的"初始等级"占比降到最低。
本项目输出：
  逐帧：ε1 各级占比 p1_0..31、ε2 各级占比 p2_0..31、ε1 中位数（仅作诊断）
  逐秒特征（过去 5 s 平均，因果）——"Dong-fix" 组：
    dong_P1_ge3   P(ε1 ≥ 2.81‰)   ≈ 原文 32 级中第 3 级以上（低阈值）
    dong_P1_ge6   P(ε1 ≥ 6.56‰)   ≈ 原文 εmd 量级（Dong 6 件试样 εmd 均值 5.4–6.7‰）
    dong_P1_ge12  P(ε1 ≥ 12.2‰)   高等级（塑性）
    dong_P2_init  ε2 初始等级（−0.3125‰..0）占比 → 压密程度
    dong_level_entropy  ε1 32 级占比分布的熵 / log(32)（分级"分散度"）
  事件：
    各阈值 θ 下 P(ε1≥θ) 首次 ≥ 2% 的时刻（Dong 判据 1 的阈值灵敏度）
    ε2 初始等级占比首次 < 2% 的时刻（Dong 判据 2 的可操作版：原文"最小值"在 HS-2 上会降到 0 并保持）
输出 outputs/dong_frame_levels.csv, outputs/samples.csv（E2 样本 + Dong-fix 列）, outputs/dong_events.json,
     outputs/feature_catalog.csv
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E2 = HERE.parent / "E2_尺度无关DIC特征聚类"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def load_by_path(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F2 = load_by_path(E2 / "01_features.py", "e2_features")
C, MF, FX = F2.C, F2.MF, F2.FX
W = 5
N_LEV = 32
E1_RANGE = (0.0, 0.03)
E2_RANGE = (-0.01, 0.0)
LV1 = np.linspace(*E1_RANGE, N_LEV + 1)
LV2 = np.linspace(*E2_RANGE, N_LEV + 1)
THETAS = [LV1[j] for j in range(2, 14)]           # 1.88‰ … 12.2‰，Dong 判据 1 的阈值灵敏度
SEL = {"dong_P1_ge3": 3, "dong_P1_ge6": 7, "dong_P1_ge12": 13}   # 级号 j → 阈值 LV1[j]


def frame_levels():
    path = OUT / "dong_frame_levels.csv"
    if path.exists():
        return pd.read_csv(path)
    a = C.open_cache()
    rows = []
    for k in range(1, C.N_FRAMES):
        f, ok, _ = F2.frame_fields(a, k)
        if ok.sum() < 1000:
            rows.append({"frame": k})
            continue
        _, e1, e2 = F2.principal(f)
        e1, e2 = e1[ok], e2[ok]
        h1 = np.histogram(np.clip(e1, LV1[0], LV1[-1] - 1e-12), LV1)[0] / len(e1)
        h2 = np.histogram(np.clip(e2, LV2[0] + 1e-12, LV2[-1] - 1e-12), LV2)[0] / len(e2)
        r = {"frame": k, "e1_med": float(np.median(e1)), "e2_med": float(np.median(e2))}
        r.update({f"p1_{i}": h1[i] for i in range(N_LEV)})
        r.update({f"p2_{i}": h2[i] for i in range(N_LEV)})
        rows.append(r)
        if k % 250 == 0:
            print(f"\r  Dong 分级 帧 {k}/{C.N_FRAMES}", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def derived(df):
    p1 = df[[f"p1_{i}" for i in range(N_LEV)]].to_numpy()
    out = pd.DataFrame({"frame": df.frame})
    for nm, j in SEL.items():
        out[nm] = p1[:, j:].sum(axis=1)
    out["dong_P2_init"] = df[f"p2_{N_LEV - 1}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p1 > 0, p1 * np.log(p1), 0), axis=1) / np.log(N_LEV)
    out["dong_level_entropy"] = ent
    return out


def events(df, fi):
    t = df.merge(fi, on="frame").t_s.to_numpy()
    p1 = df[[f"p1_{i}" for i in range(N_LEV)]].to_numpy()
    ev = {"criterion1_threshold_scan": []}
    for j in range(2, 14):
        P = p1[:, j:].sum(axis=1)
        sm = pd.Series(P).rolling(5, min_periods=1).median().to_numpy()
        idx = np.where(sm >= 0.02)[0]
        ev["criterion1_threshold_scan"].append(dict(level=j, theta_permil=round(LV1[j] * 1e3, 3),
                                                    t_cross_2pct=float(t[idx[0]]) if len(idx) else None))
    p2i = pd.Series(df[f"p2_{N_LEV - 1}"].to_numpy()).rolling(25, min_periods=5).median().to_numpy()
    idx = np.where((p2i < 0.02) & (t > 10))[0]
    ev["criterion2_e2_init_below_2pct_s"] = float(t[idx[0]]) if len(idx) else None
    pre = t < 427
    ev["criterion2_e2_init_argmin_prepeak_s"] = float(t[pre][np.nanargmin(p2i[pre])])
    ev["note"] = ("判据 1 原文需人工判读'突增'；此处给出各阈值首次 ≥2% 的时刻。判据 2 原文取'最小值'，"
                  "HS-2 上该占比在约 150 s 后降到 0 并长期保持，argmin 无确定意义，故同时给出'首次 < 2%'。")
    return ev


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = frame_levels()
    fi = pd.read_csv(C.OUT / "frame_index.csv")[["frame", "t_s"]]
    der = derived(df).merge(fi, on="frame")
    der = der[der.t_s >= 0]
    der["sec"] = np.floor(der.t_s).astype(int)
    s = pd.read_csv(E2 / "outputs" / "samples.csv")
    per = der.groupby("sec").mean(numeric_only=True).reindex(s.sec)
    dong_cols = list(SEL) + ["dong_P2_init", "dong_level_entropy"]
    for c in dong_cols:
        s[c] = FX.past(per[c].reset_index(drop=True), W).to_numpy()
    s.to_csv(OUT / "samples.csv", index=False, encoding="utf-8-sig")
    cat = pd.read_csv(E2 / "outputs" / "feature_catalog.csv")
    mean = {"dong_P1_ge3": f"P(ε1 ≥ {LV1[3] * 1e3:.2f}‰)（固定量程第 4 级以上）",
            "dong_P1_ge6": f"P(ε1 ≥ {LV1[7] * 1e3:.2f}‰)（≈ Dong εmd 量级）",
            "dong_P1_ge12": f"P(ε1 ≥ {LV1[13] * 1e3:.2f}‰)（高等级）",
            "dong_P2_init": "ε2 初始等级（−0.31‰..0）占比（压密）",
            "dong_level_entropy": "ε1 32 级占比分布的归一化熵"}
    cat = pd.concat([cat, pd.DataFrame([dict(feature=c, modality="DIC", kind="Dong固定量程",
                                             ren_section="Dong SCCM 原文", meaning=mean[c], group_zh="Dong固定量程")
                                        for c in dong_cols])], ignore_index=True)
    cat.to_csv(OUT / "feature_catalog.csv", index=False, encoding="utf-8-sig")
    ev = events(df.dropna(subset=["p1_0"]), fi)
    (OUT / "dong_events.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Dong 判据 1（各阈值首次 ≥2%）：")
    for r in ev["criterion1_threshold_scan"]:
        print(f"  θ = {r['theta_permil']:6.2f}‰ → {r['t_cross_2pct']}")
    print("Dong 判据 2：ε2 初始等级首次 < 2% 于", ev["criterion2_e2_init_below_2pct_s"], "s；argmin 于",
          ev["criterion2_e2_init_argmin_prepeak_s"], "s")


if __name__ == "__main__":
    main()
