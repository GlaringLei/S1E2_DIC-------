"""02 单模态聚类基线：AE / DIC 特征分别聚类，与载荷人工阶段比对。

评价：ARI 为主（聚类 vs 人工 4 阶段，只在阶段明确的秒上计算），轮廓系数为辅（聚类在自身 PCA 空间的紧致/分离度）。
另报：NMI、轮廓系数(人工标签)——同一空间里人工阶段本身有多可分、逐阶段纯度、聚类在时间上的碎片化程度。

聚类基线（k = 4，与人工阶段数一致；另扫 k = 2..8 看 ARI/轮廓系数随 k 的变化）
  KMeans          —— 球形簇，最常用基线
  GMM（全协方差） —— 椭球簇、软划分
  Ward 层次聚类   —— 自下而上合并
  KMeans+时间约束（参考）—— 在特征后拼接按特征尺度放大的时间列，展示"加时序先验"能到哪；不计入主比较
基线：仅时间（把 sec 按 KMeans 切 4 段），说明单试样阶段与时间一一对应的上限。

稳定性：KMeans / GMM 各 20 个随机种子，报告 ARI 均值 ± 标准差。
输出 outputs/cluster_labels.csv, outputs/metrics_main.csv, outputs/metrics_k_scan.csv,
     outputs/contingency_*.csv, outputs/stability.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
K = 4
SEEDS = range(20)
STAGE_ZH = ["起裂前", "起裂至损伤", "损伤至峰值", "峰后"]
SETS = [("AE", "全部"), ("AE", "去趋势"), ("DIC", "全部"), ("DIC", "去趋势")]


def load():
    s = pd.read_csv(OUT / "samples.csv")
    emb = {k: pd.read_csv(OUT / f"embed_{k[0]}_{k[1]}.csv") for k in SETS}
    return s, emb


def cluster(E, method, k=K, seed=0):
    if method == "KMeans":
        return KMeans(k, n_init=20, random_state=seed).fit_predict(E)
    if method == "GMM":
        return GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed, reg_covar=1e-4).fit_predict(E)
    if method == "Ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(E)
    raise ValueError(method)


def align_to_stages(y, c):
    """匈牙利算法把簇号映射到最匹配的阶段号（仅用于作图配色与纯度，不影响 ARI）。"""
    ks = np.unique(c)
    M = np.array([[np.sum((c == a) & (y == s)) for s in range(4)] for a in ks])
    r, col = linear_sum_assignment(-M)
    mp = {ks[i]: col[j] for j, i in enumerate(r)} if len(ks) <= 4 else {}
    rest = [a for a in ks if a not in mp]
    for a in rest:                               # k > 4 时多余簇映射到占比最大的阶段 + 10
        mp[a] = 10 + int(np.argmax(M[list(ks).index(a)]))
    return np.array([mp[a] for a in c])


def fragmentation(c_sorted):
    """时间上相邻两秒簇号不同的次数（真实阶段只有 3 次切换）。"""
    return int(np.sum(c_sorted[1:] != c_sorted[:-1]))


def evaluate(y, c, E, mask):
    yc, cc, Ec = y[mask], c[mask], E[mask]
    sil = silhouette_score(Ec, cc) if len(np.unique(cc)) > 1 else np.nan
    return dict(ARI=adjusted_rand_score(yc, cc), NMI=normalized_mutual_info_score(yc, cc),
                silhouette=sil, n_switches=fragmentation(c))


def group_ablation(s, mask):
    """按特征组（强度/状态、机制、动态、累积）分别聚类，定位哪一组决定了模态的可分性。
    预处理与 01 相同（中位数填补 → 鲁棒标准化 → 截尾 → PCA 90%）。"""
    import importlib
    pre = importlib.import_module("01_features").preprocess
    cat = pd.read_csv(OUT / "feature_catalog.csv")
    y = s.stage_code.to_numpy()
    rows = []
    for mod in ("AE", "DIC"):
        c = cat[cat.modality == mod]
        groups = {g: list(c[c.group_zh == g].feature) for g in c.group_zh.unique()}
        groups["全部"] = list(c.feature)
        if mod == "AE":
            groups["强度+动态"] = groups["强度/状态"] + groups["动态"]
        for g, cols in groups.items():
            _, E, _, _ = pre(s[cols])
            for method in ("KMeans", "GMM", "Ward"):
                aris = [adjusted_rand_score(y[mask], cluster(E, method, seed=sd)[mask])
                        for sd in (range(5) if method != "Ward" else [0])]
                cc = cluster(E, method)
                sil = silhouette_score(E[mask], cc[mask]) if E.shape[1] > 0 and len(np.unique(cc)) > 1 else np.nan
                rows.append(dict(modality=mod, group=g, n_features=len(cols), n_pcs=E.shape[1], method=method,
                                 ARI=float(np.mean(aris)), silhouette=sil,
                                 silhouette_true_stage=silhouette_score(E[mask], y[mask])))
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    s, emb = load()
    y_all = s.stage_code.to_numpy()
    mask = (y_all >= 0) & (s.uncertain.to_numpy() == 0)
    y = y_all.copy()

    # 人工阶段在各特征空间中的轮廓系数（可分性上限参考）
    ref = []
    for k, e in emb.items():
        E = e.filter(like="PC").to_numpy()
        ref.append(dict(modality=k[0], feature_set=k[1],
                        silhouette_true_stage=silhouette_score(E[mask], y[mask])))
    ref = pd.DataFrame(ref)

    rows, labels, stab = [], {"sec": s.sec, "stage_code": y_all, "uncertain": s.uncertain}, []
    for (mod, tag), e in emb.items():
        E = e.filter(like="PC").to_numpy()
        for method in ("KMeans", "GMM", "Ward"):
            c = cluster(E, method)
            labels[f"{mod}_{tag}_{method}"] = align_to_stages(np.where(mask, y, -1), c)
            r = evaluate(y, c, E, mask)
            rows.append(dict(modality=mod, feature_set=tag, method=method, n_pcs=E.shape[1], **r))
            if method in ("KMeans", "GMM"):
                aris = [adjusted_rand_score(y[mask], cluster(E, method, seed=sd)[mask]) for sd in SEEDS]
                stab.append(dict(modality=mod, feature_set=tag, method=method,
                                 ARI_mean=np.mean(aris), ARI_std=np.std(aris), ARI_min=np.min(aris), ARI_max=np.max(aris)))
        # 参考：加时间先验
        t = (s.sec.to_numpy() - s.sec.mean()) / s.sec.std() * np.sqrt(E.shape[1])
        Et = np.c_[E, t]
        c = cluster(Et, "KMeans")
        r = evaluate(y, c, E, mask)
        rows.append(dict(modality=mod, feature_set=tag, method="KMeans+时间(参考)", n_pcs=E.shape[1], **r))
        labels[f"{mod}_{tag}_KMeans+时间"] = align_to_stages(np.where(mask, y, -1), c)
    # 仅时间基线
    Tt = s[["sec"]].to_numpy(float)
    c = cluster(Tt, "KMeans")
    rows.append(dict(modality="时间", feature_set="基线", method="KMeans", n_pcs=1, **evaluate(y, c, Tt, mask)))
    labels["时间_基线_KMeans"] = align_to_stages(np.where(mask, y, -1), c)

    m = pd.DataFrame(rows).merge(ref, on=["modality", "feature_set"], how="left")
    m.to_csv(OUT / "metrics_main.csv", index=False, encoding="utf-8-sig")
    lab = pd.DataFrame(labels)
    lab.to_csv(OUT / "cluster_labels.csv", index=False, encoding="utf-8-sig")
    st = pd.DataFrame(stab)
    st.to_csv(OUT / "stability.csv", index=False, encoding="utf-8-sig")

    # 列联表与逐阶段纯度（KMeans、全部特征）
    for mod in ("AE", "DIC"):
        for method in ("KMeans", "GMM", "Ward"):
            c = lab[f"{mod}_全部_{method}"].to_numpy()
            ct = pd.crosstab(pd.Series(y[mask], name="人工阶段").map(dict(enumerate(STAGE_ZH))),
                             pd.Series(c[mask], name="簇（已对齐）"))
            ct = ct.reindex(STAGE_ZH)
            ct.to_csv(OUT / f"contingency_{mod}_{method}.csv", encoding="utf-8-sig")

    # k 扫描
    ks = []
    for (mod, tag), e in emb.items():
        E = e.filter(like="PC").to_numpy()
        for k in range(2, 9):
            for method in ("KMeans", "GMM", "Ward"):
                c = cluster(E, method, k=k)
                ks.append(dict(modality=mod, feature_set=tag, method=method, k=k, **evaluate(y, c, E, mask)))
    ks = pd.DataFrame(ks)
    ks.to_csv(OUT / "metrics_k_scan.csv", index=False, encoding="utf-8-sig")

    print("== 人工阶段在各特征空间中的轮廓系数（可分性参考） ==")
    print(ref.round(3).to_string(index=False))
    print("\n== 主结果（k=4；ARI 为主，轮廓系数为辅） ==")
    print(m.drop(columns=["silhouette_true_stage"]).round(3).to_string(index=False))
    print("\n== 稳定性（20 个随机种子） ==")
    print(st.round(3).to_string(index=False))
    print("\n== 列联表：KMeans，全部特征 ==")
    for mod in ("AE", "DIC"):
        print(f"[{mod}]\n", pd.read_csv(OUT / f"contingency_{mod}_KMeans.csv", index_col=0).to_string())
    print("\n== k 扫描：各 k 下 ARI 最高的方法 ==")
    best = ks.loc[ks.groupby(["modality", "feature_set", "k"]).ARI.idxmax()]
    print(best.pivot_table(index="k", columns=["modality", "feature_set"], values="ARI").round(3).to_string())
    print("\n  轮廓系数（KMeans）随 k：")
    print(ks[ks.method == "KMeans"].pivot_table(index="k", columns=["modality", "feature_set"],
                                                 values="silhouette").round(3).to_string())

    ab = group_ablation(s, mask)
    ab.to_csv(OUT / "metrics_group_ablation.csv", index=False, encoding="utf-8-sig")
    print("\n== 特征组消融（ARI，KMeans/GMM 为 5 个种子平均） ==")
    print(ab.pivot_table(index=["modality", "group", "n_features", "silhouette_true_stage"], columns="method",
                         values="ARI").round(3).to_string())


if __name__ == "__main__":
    main()
