"""01 AE / DIC 特征提取与预处理（单模态聚类基线的输入）。

两种特征提取方法（均来自 Ren et al. 2026，见 ../Ren2026/00_Ren方法伪代码.md）：

  AE-Ren  —— 每 1 s 一个样本，统计过去 W=5 s 的 AE 撞击：
      强度  ae_rate / ae_ring_rate / ae_energy_rate            （§6 事件率）
      机制  ae_tensile_rate / ae_shear_rate / ae_shear_frac     （§6 RA–AF 拉伸/剪切分类）
            ae_RA_med / ae_AF_med / ae_amp_med / ae_dur_med     （§6 波形参数）
      动态  ae_zero_frac / ae_rate_slope / ae_rate_cv            （§7 平静程度、突发性）
      累积  ae_cum_hits                                          （§7 累计撞击）
  DIC-Ren —— 每 1 s 一个样本，统计过去 W=5 s 的 DIC 帧：
      带运动学  dic_dn / dic_ds / dic_dmag（累积），及其速率、加速比、断面不同步度（§3–§4）
      能量应力  dic_U / dic_tau 及变化率（§5）
      场统计    dic_gmax_p50 / dic_gmax_lift / 局部化强度变化率（§1）、dic_crack、dic_eyy（补充）

两套特征集：
  全部    —— 方法定义的全部特征
  去趋势  —— 去掉"累积"类（随加载单调增长，单试样上近似时钟），用作消融

预处理（每个模态、每套特征集独立做，不使用阶段标签）：
  1 缺失值：列中位数填补（主要是窗口内撞击过少时 RA/AF/剪切占比无定义）
  2 鲁棒标准化：(x − 中位数) / (IQR/1.349)；IQR=0 时退回标准差，仍为 0 则删除该列
  3 截尾到 ±5，避免峰后极端值主导距离
  4 PCA 保留 90% 方差 → 聚类空间（轮廓系数也在此空间计算）

复用 ../Ren2026/outputs/ae_seconds.csv、dic_seconds.csv 与 ../Ren2026/02_extract_features.py 的特征函数。
输出 outputs/samples.csv, outputs/feature_catalog.csv, outputs/embed_{模态}_{特征集}.csv, outputs/preprocess.json
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
REN = HERE.parent / "Ren2026"
sys.path.insert(0, str(REN))
FX = importlib.import_module("02_extract_features")   # 同时把 methodScripts 加入路径
import common as C                                    # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
W = 5
PCA_VAR = 0.90
CLIP = 5.0
GROUP_ZH = {"state": "强度/状态", "mech": "机制", "dynamic": "动态", "trend": "累积"}


def extract():
    ae = pd.read_csv(REN / "outputs" / "ae_seconds.csv")
    dic = pd.read_csv(REN / "outputs" / "dic_seconds.csv")
    dic = dic[dic.sec.isin(ae.sec)].reset_index(drop=True)
    FX.CATALOG.clear()
    fa = FX.ae_features(ae, W)
    fd = FX.dic_features(dic, W)
    cat = pd.DataFrame(FX.CATALOG)
    cat["group_zh"] = cat.kind.map(GROUP_ZH)
    df = ae[["sec", "stage_code", "uncertain", "load_kN"]].merge(fa, on="sec").merge(fd, on="sec", how="left")
    return df, cat


def preprocess(X: pd.DataFrame):
    info = {}
    X = X.copy()
    nan_frac = X.isna().mean()
    X = X.fillna(X.median())
    keep = []
    for c in X.columns:
        med = X[c].median()
        q1, q3 = X[c].quantile([0.25, 0.75])
        s = (q3 - q1) / 1.349
        how = "IQR"
        if s <= 0:
            s, how = X[c].std(), "std"
        if not np.isfinite(s) or s <= 0:
            info[c] = dict(dropped=True)
            continue
        X[c] = ((X[c] - med) / s).clip(-CLIP, CLIP)
        keep.append(c)
        info[c] = dict(nan_frac=float(nan_frac[c]), center=float(med), scale=float(s), scale_from=how,
                       clipped_frac=float((X[c].abs() >= CLIP).mean()))
    Z = X[keep].to_numpy(float)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=0).fit(Z)
    E = pca.transform(Z)
    return X[keep], E, pca, info


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, cat = extract()
    df.to_csv(OUT / "samples.csv", index=False, encoding="utf-8-sig")
    cat.to_csv(OUT / "feature_catalog.csv", index=False, encoding="utf-8-sig")
    log = {"W_s": W, "pca_var": PCA_VAR, "clip": CLIP, "n_samples": len(df),
           "n_certain": int(((df.stage_code >= 0) & (df.uncertain == 0)).sum()), "sets": {}}
    for mod in ("AE", "DIC"):
        for tag in ("全部", "去趋势"):
            c = cat[(cat.modality == mod) & ((cat.kind != "trend") if tag == "去趋势" else True)]
            Xs, E, pca, info = preprocess(df[list(c.feature)])
            emb = pd.DataFrame(E, columns=[f"PC{i + 1}" for i in range(E.shape[1])])
            emb.insert(0, "sec", df.sec)
            emb.to_csv(OUT / f"embed_{mod}_{tag}.csv", index=False, encoding="utf-8-sig")
            if tag == "全部":
                Xs.insert(0, "sec", df.sec)
                Xs.to_csv(OUT / f"scaled_{mod}.csv", index=False, encoding="utf-8-sig")
            load = pd.DataFrame(pca.components_[:2].T, index=Xs.columns.drop("sec", errors="ignore"),
                                columns=["PC1", "PC2"])
            log["sets"][f"{mod}_{tag}"] = dict(
                n_features=len(c), kept=[k for k, v in info.items() if not v.get("dropped")],
                n_pcs=int(E.shape[1]), explained=[round(float(v), 4) for v in pca.explained_variance_ratio_],
                top_PC1=load.PC1.abs().sort_values(ascending=False).head(5).round(3).to_dict(),
                top_PC2=load.PC2.abs().sort_values(ascending=False).head(5).round(3).to_dict(),
                features=info)
            print(f"{mod:3s} {tag}: {len(c):2d} 特征 → {E.shape[1]} 个主成分（累计方差 "
                  f"{pca.explained_variance_ratio_.sum():.2f}）；PC1 主要载荷: "
                  + ", ".join(f"{k}" for k in load.PC1.abs().sort_values(ascending=False).head(3).index))
    (OUT / "preprocess.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"样本 {log['n_samples']} 秒；评估用（阶段明确）{log['n_certain']} 秒")


if __name__ == "__main__":
    main()
