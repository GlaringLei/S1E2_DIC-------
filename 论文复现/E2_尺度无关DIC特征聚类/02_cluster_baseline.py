"""02 E2 单模态聚类基线：与 E1 相同的评价流程（ARI 为主、轮廓系数为辅），并把 E1 结果并排对照。

在 E1 的基础上固定加入（E1 §7 的建议）：
  - 零模型百分位：把 0–457 s 随机切成 4 个连续段（2 万次），观测 ARI 在其中的百分位
  - 边界偏差：聚类切换点与人工边界（起裂 / 损伤 / 峰值）的距离
  - 特征与时间的 |ρ|：衡量特征集携带了多少"时钟"
  - 时间外推：GMM 只用前 T 秒的样本拟合，再给全程样本打标签（T = 300 / 350 / 400 s），
    检验在"没见过的时间段"上能否识别新状态（单试样内最接近跨试样的检验）
输出 outputs/cluster_labels.csv, metrics_main.csv, metrics_k_scan.csv, metrics_group_ablation.csv,
     stability.csv, contingency_*.csv, null_model.csv, boundary_offset.csv, extrapolation.csv, compare_E1.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "E1_单模态聚类基线"
OUT = HERE / "outputs"


def load_by_path(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CB = load_by_path(E1 / "02_cluster_baseline.py", "e1_cluster")      # cluster / align_to_stages / evaluate
F01 = load_by_path(HERE / "01_features.py", "e2_features")
PRE = F01.E1F.preprocess
K = 4
SEEDS = range(20)
N_NULL = 20000
STAGE_ZH = ["起裂前", "起裂至损伤", "损伤至峰值", "峰后"]
SETS = [("AE", "全部"), ("AE", "去趋势"), ("DIC", "全部"), ("DIC", "累积场"), ("DIC", "增量动态")]
BOUNDS = np.array([270.0, 334.0, 427.3])


def contiguous_null(n, idx, y, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(N_NULL)
    for i in range(N_NULL):
        b = np.sort(rng.choice(np.arange(1, n), 3, replace=False))
        out[i] = adjusted_rand_score(y[idx], np.searchsorted(b, np.arange(n), side="right")[idx])
    return out


def boundary_offset(sec, c):
    sw = sec[np.where(np.diff(c) != 0)[0] + 1]
    if not len(sw):
        return [np.nan] * 3
    return [float(sw[np.argmin(np.abs(sw - b))] - b) for b in BOUNDS]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 230)
    s = pd.read_csv(OUT / "samples.csv")
    cat = pd.read_csv(OUT / "feature_catalog.csv")
    emb = {k: pd.read_csv(OUT / f"embed_{k[0]}_{k[1]}.csv") for k in SETS}
    y_all = s.stage_code.to_numpy()
    mask = (y_all >= 0) & (s.uncertain.to_numpy() == 0)
    y = y_all
    sec = s.sec.to_numpy()
    null = contiguous_null(len(s), np.where(mask)[0], y)
    pd.Series(null).describe(percentiles=[.5, .9, .95, .99]).to_csv(OUT / "null_model.csv", encoding="utf-8-sig")
    pct = lambda a: float((null < a).mean() * 100)

    rho = {f: abs(spearmanr(s[[f, "sec"]].dropna()[f], s[[f, "sec"]].dropna().sec)[0]) for f in cat.feature}
    set_cols = {("AE", "全部"): list(cat[cat.modality == "AE"].feature),
                ("AE", "去趋势"): list(cat[(cat.modality == "AE") & (cat.kind != "trend")].feature),
                ("DIC", "全部"): list(cat[cat.modality == "DIC"].feature),
                ("DIC", "累积场"): F01.GROUPS["累积场形态"],
                ("DIC", "增量动态"): F01.GROUPS["增量场形态"] + F01.GROUPS["演化动态"]}

    ref, rows, stab, offs = [], [], [], []
    labels = {"sec": s.sec, "stage_code": y_all, "uncertain": s.uncertain}
    for (mod, tag), e in emb.items():
        E = e.filter(like="PC").to_numpy()
        ref.append(dict(modality=mod, feature_set=tag, silhouette_true_stage=silhouette_score(E[mask], y[mask]),
                        median_abs_rho_time=float(np.median([rho[f] for f in set_cols[(mod, tag)]]))))
        for method in ("KMeans", "GMM", "Ward"):
            c = CB.cluster(E, method)
            labels[f"{mod}_{tag}_{method}"] = CB.align_to_stages(np.where(mask, y, -1), c)
            r = CB.evaluate(y, c, E, mask)
            off = boundary_offset(sec, c)
            rows.append(dict(modality=mod, feature_set=tag, method=method, n_pcs=E.shape[1], **r,
                             null_percentile=pct(r["ARI"]), off_ci_s=off[0], off_cd_s=off[1], off_peak_s=off[2]))
            if method != "Ward":
                aris = [adjusted_rand_score(y[mask], CB.cluster(E, method, seed=sd)[mask]) for sd in SEEDS]
                stab.append(dict(modality=mod, feature_set=tag, method=method, ARI_mean=np.mean(aris),
                                 ARI_std=np.std(aris), ARI_min=np.min(aris), ARI_max=np.max(aris)))
    Tt = s[["sec"]].to_numpy(float)
    c = CB.cluster(Tt, "KMeans")
    r = CB.evaluate(y, c, Tt, mask)
    off = boundary_offset(sec, c)
    rows.append(dict(modality="时间", feature_set="基线", method="KMeans", n_pcs=1, **r, null_percentile=pct(r["ARI"]),
                     off_ci_s=off[0], off_cd_s=off[1], off_peak_s=off[2]))
    labels["时间_基线_KMeans"] = CB.align_to_stages(np.where(mask, y, -1), c)
    m = pd.DataFrame(rows).merge(pd.DataFrame(ref), on=["modality", "feature_set"], how="left")
    m.to_csv(OUT / "metrics_main.csv", index=False, encoding="utf-8-sig")
    lab = pd.DataFrame(labels)
    lab.to_csv(OUT / "cluster_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stab).to_csv(OUT / "stability.csv", index=False, encoding="utf-8-sig")

    for mod in ("AE", "DIC"):
        for method in ("KMeans", "GMM", "Ward"):
            cc = lab[f"{mod}_全部_{method}"].to_numpy()
            ct = pd.crosstab(pd.Series(y[mask], name="人工阶段").map(dict(enumerate(STAGE_ZH))),
                             pd.Series(cc[mask], name="簇（已对齐）")).reindex(STAGE_ZH)
            ct.to_csv(OUT / f"contingency_{mod}_{method}.csv", encoding="utf-8-sig")

    ks = []
    for (mod, tag), e in emb.items():
        E = e.filter(like="PC").to_numpy()
        for k in range(2, 9):
            for method in ("KMeans", "GMM", "Ward"):
                ks.append(dict(modality=mod, feature_set=tag, method=method, k=k,
                               **CB.evaluate(y, CB.cluster(E, method, k=k), E, mask)))
    ks = pd.DataFrame(ks)
    ks.to_csv(OUT / "metrics_k_scan.csv", index=False, encoding="utf-8-sig")

    # 特征组消融
    ab = []
    for mod in ("AE", "DIC"):
        c_ = cat[cat.modality == mod]
        groups = {g: list(c_[c_.group_zh == g].feature) for g in c_.group_zh.unique()}
        groups["全部"] = list(c_.feature)
        for g, cols in groups.items():
            _, E, _, _ = PRE(s[cols])
            for method in ("KMeans", "GMM", "Ward"):
                aris = [adjusted_rand_score(y[mask], CB.cluster(E, method, seed=sd)[mask])
                        for sd in (range(5) if method != "Ward" else [0])]
                cc = CB.cluster(E, method)
                ab.append(dict(modality=mod, group=g, n_features=len(cols), n_pcs=E.shape[1], method=method,
                               ARI=float(np.mean(aris)), null_percentile=pct(float(np.mean(aris))),
                               silhouette=silhouette_score(E[mask], cc[mask]) if len(np.unique(cc)) > 1 else np.nan,
                               silhouette_true_stage=silhouette_score(E[mask], y[mask]),
                               median_abs_rho_time=float(np.median([rho[f] for f in cols]))))
    ab = pd.DataFrame(ab)
    ab.to_csv(OUT / "metrics_group_ablation.csv", index=False, encoding="utf-8-sig")

    # 时间外推（GMM 在前 T 秒拟合，全程预测；预处理也只用前 T 秒的统计量）
    ex = []
    for mod in ("AE", "DIC"):
        cols = set_cols[(mod, "全部")]
        for T in (300, 350, 400, 457):
            fit = sec < T
            X = s[cols]
            Xf = X[fit]
            med = Xf.median()
            q1, q3 = Xf.quantile(0.25), Xf.quantile(0.75)
            sc = ((q3 - q1) / 1.349).replace(0, np.nan).fillna(Xf.std()).replace(0, 1.0)
            Z = ((X.fillna(med) - med) / sc).clip(-5, 5).to_numpy()
            from sklearn.decomposition import PCA
            pca = PCA(0.9, svd_solver="full").fit(Z[fit])
            E = pca.transform(Z)
            g = GaussianMixture(K, covariance_type="full", n_init=5, random_state=0, reg_covar=1e-4).fit(E[fit])
            c = g.predict(E)
            unseen = mask & ~fit
            ex.append(dict(modality=mod, fit_until_s=T, ARI_all=adjusted_rand_score(y[mask], c[mask]),
                           ARI_fit_part=adjusted_rand_score(y[mask & fit], c[mask & fit]) if (mask & fit).sum() else np.nan,
                           n_unseen=int(unseen.sum()),
                           stages_in_fit=int(len(np.unique(y[mask & fit]))),
                           new_cluster_share_unseen=float(np.mean([ci not in set(c[fit]) for ci in c[unseen]])) if unseen.any() else np.nan,
                           off_ci_s=boundary_offset(sec, c)[0], off_cd_s=boundary_offset(sec, c)[1],
                           off_peak_s=boundary_offset(sec, c)[2]))
    ex = pd.DataFrame(ex)
    ex.to_csv(OUT / "extrapolation.csv", index=False, encoding="utf-8-sig")

    # 与 E1 对照
    e1 = pd.read_csv(E1 / "outputs" / "metrics_main.csv")
    e1 = e1[e1.method.isin(["KMeans", "GMM", "Ward"]) & (e1.modality.isin(["AE", "DIC"]))
            & (e1.feature_set == "全部")][["modality", "method", "ARI", "silhouette", "n_switches"]]
    e2 = m[m.feature_set == "全部"][["modality", "method", "ARI", "silhouette", "n_switches", "null_percentile",
                                    "median_abs_rho_time"]]
    cmp_ = e1.merge(e2, on=["modality", "method"], suffixes=("_E1", "_E2"))
    cmp_.to_csv(OUT / "compare_E1.csv", index=False, encoding="utf-8-sig")

    q = np.percentile(null, [50, 95])
    print(f"零模型（随机连续 4 段）ARI：中位数 {q[0]:.3f}，95% 分位 {q[1]:.3f}")
    print("\n== 主结果（k=4） ==")
    print(m.round(3).to_string(index=False))
    print("\n== 稳定性 ==")
    print(pd.DataFrame(stab).round(3).to_string(index=False))
    print("\n== 列联表：DIC 全部 ==")
    for meth in ("KMeans", "GMM", "Ward"):
        print(f"[{meth}]\n", pd.read_csv(OUT / f"contingency_DIC_{meth}.csv", index_col=0).to_string())
    print("\n== k 扫描（各 k 取三方法最大 ARI） ==")
    best = ks.loc[ks.groupby(["modality", "feature_set", "k"]).ARI.idxmax()]
    print(best.pivot_table(index="k", columns=["modality", "feature_set"], values="ARI").round(3).to_string())
    print("\n== 特征组消融 ==")
    print(ab.pivot_table(index=["modality", "group", "n_features", "median_abs_rho_time", "silhouette_true_stage"],
                         columns="method", values="ARI").round(3).to_string())
    print("\n== 时间外推（GMM，全部特征） ==")
    print(ex.round(3).to_string(index=False))
    print("\n== 与 E1 对照（全部特征） ==")
    print(cmp_.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
