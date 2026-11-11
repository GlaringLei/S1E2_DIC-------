"""04 补充分析：回答 README 补充部分的问题 2（GMM 为何在 DIC 上更好）与问题 3（时间结构的影响）。

A  GMM 优势的来源（DIC·全部，PCA 空间）
   A1 人工阶段在 PCA 空间中的散布：样本数、总方差（协方差迹）、各向异性（最大/最小特征值比）
   A2 协方差类型消融：GMM full / diag / spherical / tied 与 KMeans，各 20 个种子
       spherical = 各簇大小不同但各向同性；tied = 各簇共享同一协方差（形状可拉长但大小相同）
       → 判断起作用的是"簇大小不等"还是"簇形状不同"
   A3 各方法的簇大小
B  时间结构的影响（不增加试样时能做的检查）
   B1 确认特征矩阵中没有绝对时间列
   B2 每个特征与时间的 Spearman |ρ|（全程 / 峰前）
   B3 零模型：把 0–457 s 随机切成 4 个连续段（3 个边界均匀随机），计算与人工阶段的 ARI 分布；
       观测 ARI 超过该分布的百分位 = "比随便一个连续时间分段好多少"
   B4 时间去相关特征集：剔除与时间 |ρ| > 0.9 的特征后重新聚类
   B5 非重叠采样：每 5 s 取 1 个样本（窗口不重叠），检查 ARI 是否依赖相邻样本的冗余
   B6 边界偏差：聚类在时间上的切换点与人工边界的距离
输出 outputs/supp_*.csv, figures/fig9_supplement.png
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Ellipse, Patch         # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT, FIG = HERE / "outputs", HERE / "figures"
CB = importlib.import_module("02_cluster_baseline")
PRE = importlib.import_module("01_features").preprocess
STAGE_ZH = ["起裂前", "起裂至损伤", "损伤至峰值", "峰后"]
STAGE_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEEDS = range(20)
N_NULL = 20000
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def gmm(E, cov, seed, k=4):
    return GaussianMixture(k, covariance_type=cov, n_init=5, random_state=seed, reg_covar=1e-4).fit_predict(E)


# ------------------------------------------------------------------ A
def stage_spread(E, y, mask):
    rows = []
    for s in range(4):
        X = E[mask & (y == s)]
        ev = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]
        rows.append(dict(stage=STAGE_ZH[s], n=len(X), total_var=float(ev.sum()),
                         anisotropy=float(ev[0] / max(ev[-1], 1e-12)), top_axis_share=float(ev[0] / ev.sum()),
                         centroid_PC1=float(X[:, 0].mean()), centroid_PC2=float(X[:, 1].mean())))
    return pd.DataFrame(rows)


def cov_ablation(E, y, mask):
    rows = []
    for name, fn in [("KMeans", lambda sd: KMeans(4, n_init=20, random_state=sd).fit_predict(E)),
                     ("GMM-spherical", lambda sd: gmm(E, "spherical", sd)),
                     ("GMM-diag", lambda sd: gmm(E, "diag", sd)),
                     ("GMM-tied", lambda sd: gmm(E, "tied", sd)),
                     ("GMM-full", lambda sd: gmm(E, "full", sd)),
                     ("Ward", lambda sd: AgglomerativeClustering(4, linkage="ward").fit_predict(E))]:
        seeds = [0] if name == "Ward" else SEEDS
        labs = [fn(sd) for sd in seeds]
        aris = [adjusted_rand_score(y[mask], c[mask]) for c in labs]
        sizes = sorted(np.bincount(labs[0], minlength=4).tolist(), reverse=True)
        rows.append(dict(method=name, ARI_mean=np.mean(aris), ARI_std=np.std(aris), cluster_sizes=str(sizes)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B
def contiguous_null(n, y_mask_idx, y, rng):
    """随机连续 4 段分割的 ARI 分布（边界在 1..n-1 上均匀不重复抽取）。"""
    out = np.empty(N_NULL)
    for i in range(N_NULL):
        b = np.sort(rng.choice(np.arange(1, n), 3, replace=False))
        lab = np.searchsorted(b, np.arange(n), side="right")
        out[i] = adjusted_rand_score(y[y_mask_idx], lab[y_mask_idx])
    return out


def switch_points(sec, c):
    i = np.where(np.diff(c) != 0)[0]
    return sec[i + 1]


def bic_selection(E, y, mask):
    """不看标签，用 BIC 在 协方差类型 × k 中选模型；再看所选模型的 ARI。"""
    rows = []
    for cov in ("spherical", "diag", "tied", "full"):
        for k in range(2, 9):
            g = GaussianMixture(k, covariance_type=cov, n_init=5, random_state=0, reg_covar=1e-4).fit(E)
            rows.append(dict(cov=cov, k=k, bic=g.bic(E), ARI=adjusted_rand_score(y[mask], g.predict(E)[mask])))
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    s = pd.read_csv(OUT / "samples.csv")
    cat = pd.read_csv(OUT / "feature_catalog.csv")
    lab = pd.read_csv(OUT / "cluster_labels.csv")
    m = pd.read_csv(OUT / "metrics_main.csv")
    y = s.stage_code.to_numpy()
    mask = (y >= 0) & (s.uncertain.to_numpy() == 0)
    sec = s.sec.to_numpy()
    E = pd.read_csv(OUT / "embed_DIC_全部.csv").filter(like="PC").to_numpy()

    # A
    sp = stage_spread(E, y, mask)
    ab = cov_ablation(E, y, mask)
    ab_ae = cov_ablation(pd.read_csv(OUT / "embed_AE_全部.csv").filter(like="PC").to_numpy(), y, mask)
    sp.to_csv(OUT / "supp_A1_stage_spread_DIC.csv", index=False, encoding="utf-8-sig")
    pd.concat([ab.assign(modality="DIC"), ab_ae.assign(modality="AE")]).to_csv(
        OUT / "supp_A2_cov_ablation.csv", index=False, encoding="utf-8-sig")
    print("== A1 人工阶段在 DIC PCA 空间中的散布 ==")
    print(sp.round(3).to_string(index=False))
    print("\n== A2 协方差类型消融（DIC / AE，ARI） ==")
    print(ab.round(3).to_string(index=False))
    print(ab_ae.round(3).to_string(index=False))
    bic = bic_selection(E, y, mask)
    bic.to_csv(OUT / "supp_A4_bic_selection_DIC.csv", index=False, encoding="utf-8-sig")
    best = bic.loc[bic.bic.idxmin()]
    b4 = bic[bic.k == 4].sort_values("bic")
    print(f"\n== A4 BIC 选模型（DIC，不看标签） ==\n  全局最优: {best['cov']}, k={int(best['k'])}, ARI={best['ARI']:.3f}")
    print("  k=4 时各协方差类型（BIC 越小越好）:")
    print(b4.round(3).to_string(index=False))

    # B1
    feats = list(cat.feature)
    assert "sec" not in feats and not any("time" in f for f in feats)
    print("\n== B1 特征矩阵列：", len(feats), "个，均不含 sec / 时间列 ==")

    # B2
    pre = s.stage_code.between(0, 2)
    rho = []
    for r in cat.itertuples():
        a = s[[r.feature, "sec"]].dropna()
        b = s.loc[pre, [r.feature, "sec"]].dropna()
        rho.append(dict(feature=r.feature, modality=r.modality, group=r.group_zh,
                        rho_time_all=spearmanr(a[r.feature], a.sec)[0], rho_time_prepeak=spearmanr(b[r.feature], b.sec)[0]))
    rho = pd.DataFrame(rho)
    rho["abs_all"] = rho.rho_time_all.abs()
    rho.sort_values("abs_all", ascending=False).to_csv(OUT / "supp_B2_feature_time_rho.csv", index=False, encoding="utf-8-sig")
    print("\n== B2 特征与时间的 |Spearman ρ|（中位数 / >0.9 的个数） ==")
    print(rho.groupby(["modality", "group"]).agg(median_abs_rho=("abs_all", "median"), n=("feature", "size"),
                                                 n_gt_0p9=("abs_all", lambda v: int((v > 0.9).sum()))).round(2).to_string())

    # B3
    rng = np.random.default_rng(0)
    null = contiguous_null(len(s), np.where(mask)[0], y, rng)
    obs = m[m.method.isin(["KMeans", "GMM", "Ward"]) | (m.modality == "时间")][
        ["modality", "feature_set", "method", "ARI"]].copy()
    obs["null_percentile"] = [float((null < a).mean() * 100) for a in obs.ARI]
    obs["p_value"] = [float((null >= a).mean()) for a in obs.ARI]
    obs.to_csv(OUT / "supp_B3_contiguous_null.csv", index=False, encoding="utf-8-sig")
    q = np.percentile(null, [50, 95, 99])
    print(f"\n== B3 随机连续 4 段分割的 ARI：中位数 {q[0]:.3f}，95% 分位 {q[1]:.3f}，99% 分位 {q[2]:.3f}，最大 {null.max():.3f} ==")
    print(obs.round(4).to_string(index=False))

    # B4 / B5
    rows = []
    for mod in ("AE", "DIC"):
        c_all = cat[cat.modality == mod]
        for tag, cols in [("全部", list(c_all.feature)),
                          ("去趋势", list(c_all[c_all.kind != "trend"].feature)),
                          ("时间去相关(|ρ|≤0.9)", list(rho[(rho.modality == mod) & (rho.abs_all <= 0.9)].feature)),
                          ("时间去相关(|ρ|≤0.7)", list(rho[(rho.modality == mod) & (rho.abs_all <= 0.7)].feature)),
                          ("仅动态组", list(c_all[c_all.kind == "dynamic"].feature))]:
            _, Ez, _, _ = PRE(s[cols])
            for meth in ("KMeans", "GMM", "Ward"):
                c = CB.cluster(Ez, meth)
                aris = [adjusted_rand_score(y[mask], CB.cluster(Ez, meth, seed=sd)[mask])
                        for sd in (range(5) if meth != "Ward" else [0])]
                # 非重叠：每 5 s 取 1 个（另行聚类）
                sub = np.arange(0, len(s), 5)
                cs = CB.cluster(Ez[sub], meth)
                ms = mask[sub]
                rows.append(dict(modality=mod, feature_set=tag, n_features=len(cols),
                                 median_abs_rho_time=float(rho.set_index("feature").loc[cols, "abs_all"].median()),
                                 method=meth, ARI=float(np.mean(aris)),
                                 ARI_nonoverlap_5s=adjusted_rand_score(y[sub][ms], cs[ms]),
                                 null_percentile=float((null < np.mean(aris)).mean() * 100)))
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT / "supp_B4_time_decorrelated.csv", index=False, encoding="utf-8-sig")
    print("\n== B4/B5 时间去相关特征集与非重叠采样 ==")
    print(dec.round(3).to_string(index=False))

    # B6
    bnd = np.array([270.0, 334.0, 427.3])
    rows = []
    for col in ("DIC_全部_GMM", "DIC_全部_Ward", "DIC_全部_KMeans", "时间_基线_KMeans"):
        sw = switch_points(sec, lab[col].to_numpy())
        near = [float(sw[np.argmin(np.abs(sw - b))] - b) if len(sw) else np.nan for b in bnd]
        rows.append(dict(labels=col, n_switches=len(sw), switches=str(sw.tolist()[:12]),
                         offset_ci_s=near[0], offset_cd_s=near[1], offset_peak_s=near[2]))
    bo = pd.DataFrame(rows)
    bo.to_csv(OUT / "supp_B6_boundary_offset.csv", index=False, encoding="utf-8-sig")
    print("\n== B6 聚类切换点与人工边界的偏差（最近切换点 − 人工边界，秒） ==")
    print(bo.to_string(index=False))

    plot(E, y, mask, sp, ab, null, obs, dec, rho)


def plot(E, y, mask, sp, ab, null, obs, dec, rho):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.6))
    # (a) 阶段协方差椭圆
    ax = axs[0, 0]
    for s_ in range(4):
        X = E[mask & (y == s_)][:, :2]
        ax.scatter(X[:, 0], X[:, 1], s=10, color=STAGE_C[s_], alpha=0.6, lw=0)
        cov = np.cov(X.T)
        w, V = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        for k in (1, 2):
            ax.add_patch(Ellipse(X.mean(0), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang,
                                 fill=False, ec=STAGE_C[s_], lw=1.6 if k == 2 else 1.0, ls="-" if k == 2 else ":"))
    ax.set_title("(a) DIC：人工阶段在 PC1–PC2 上的分布与协方差椭圆（1σ 虚线 / 2σ 实线）", fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.legend([Patch(color=c) for c in STAGE_C],
              [f"{STAGE_ZH[i]}  n={int(sp.n[i])}  方差={sp.total_var[i]:.2f}" for i in range(4)],
              frameon=False, fontsize=7, loc="upper right")
    # (b) 协方差类型消融
    ax = axs[0, 1]
    x = np.arange(len(ab))
    cols = ["#8a8984", "#86b6ef", "#5598e7", "#c3c2b7", "#2a78d6", "#1baf7a"]
    ax.bar(x, ab.ARI_mean, yerr=ab.ARI_std, color=cols, capsize=3, error_kw=dict(ecolor=MUTED, lw=0.8))
    for i, r in enumerate(ab.itertuples()):
        ax.text(i, r.ARI_mean + r.ARI_std + 0.02, f"{r.ARI_mean:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [m.replace("GMM-", "GMM\n") for m in ab.method], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("(b) DIC：聚类方法 / GMM 协方差类型对 ARI 的影响（20 个种子，均值±标准差）", fontsize=9)
    # (c) 连续分段零模型
    ax = axs[1, 0]
    ax.hist(null, bins=80, color="#c3c2b7", edgecolor="white", lw=0.3)
    ax.axvline(np.percentile(null, 95), color=MUTED, ls="--", lw=1)
    ax.text(np.percentile(null, 95), ax.get_ylim()[1] * 0.92, " 95% 分位", fontsize=7, color=MUTED)
    show = [("DIC", "全部", "GMM", "#eb6834"), ("DIC", "全部", "KMeans", "#f2a27f"),
            ("时间", "基线", "KMeans", "#52514e"), ("AE", "全部", "GMM", "#2a78d6")]
    for mod, tag, meth, col in show:
        r = obs[(obs.modality == mod) & (obs.feature_set == tag) & (obs.method == meth)].iloc[0]
        ax.axvline(r.ARI, color=col, lw=2)
        ax.text(r.ARI, ax.get_ylim()[1] * (0.75 if mod != "时间" else 0.55),
                f" {mod}·{meth}\n ARI {r.ARI:.2f}\n 超过 {r.null_percentile:.1f}%", fontsize=7, color=INK)
    ax.set_xlabel("ARI", fontsize=8)
    ax.set_title("(c) 零模型：把 0–457 s 随机切成 4 个连续段，与人工阶段的 ARI 分布（2 万次）", fontsize=9)
    # (d) 时间去相关
    ax = axs[1, 1]
    d = dec[dec.method == "GMM"]
    sets = list(dict.fromkeys(d.feature_set))
    x = np.arange(len(sets))
    for j, (mod, col) in enumerate((("AE", "#2a78d6"), ("DIC", "#eb6834"))):
        dd = d[d.modality == mod].set_index("feature_set").loc[sets]
        ax.bar(x + (j - 0.5) * 0.36, dd.ARI, width=0.34, color=col, label=f"{mod} · GMM")
        for i, (v, nf, rr) in enumerate(zip(dd.ARI, dd.n_features, dd.median_abs_rho_time)):
            ax.text(x[i] + (j - 0.5) * 0.36, max(v, 0) + 0.02, f"{v:.2f}\n{nf}维\n|ρ|{rr:.2f}", ha="center", fontsize=6.5)
    ax.axhline(np.percentile(null, 95), color=MUTED, ls="--", lw=1)
    ax.text(len(sets) - 0.5, np.percentile(null, 95) + 0.01, "零模型 95% 分位", fontsize=7, color=MUTED, ha="right")
    ax.set_xticks(x, sets, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(d) 逐步去掉与时间强相关的特征后的 ARI（GMM；标注 维数 与 特征–时间 |ρ| 中位数）", fontsize=9)
    for a in axs.ravel():
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.suptitle("图 9  补充分析：GMM 的优势来源与时间结构的影响", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig9_supplement.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
