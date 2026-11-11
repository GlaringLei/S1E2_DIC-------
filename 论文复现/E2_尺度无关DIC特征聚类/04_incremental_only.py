"""04 仅用 DIC 增量场 6 个特征的表征与聚类（补充实验，对应 README §6.1–6.2）。

特征：inc_P2, inc_P4, inc_q98, inc_top5, inc_nu, inc_corr（定义见 01_features.py 与 README §6.2）。
预处理、聚类、评价与 02 完全相同（复用 E1 的 preprocess / cluster / evaluate 与 E2 的零模型、边界偏差）。

输出 outputs/embed_DIC_增量场6.csv, scaled_DIC_增量场6.csv, inc6_metrics.csv, inc6_stability.csv,
     inc6_cluster_labels.csv, inc6_contingency_*.csv, inc6_stage_medians.csv, inc6_noise_reference.json
图   figures/fig10_inc6_embedding_stage.png   样本分布（对应图 3）
     figures/fig11_inc6_embedding_cluster.png 聚类结果与列联表（对应图 4、图 7）
     figures/fig12_inc6_cluster_timeline.png  聚类标签沿时间展开（对应图 5）
     figures/fig13_inc6_feature_construction.png  增量场相对分级图与 6 个特征的时间曲线
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "E1_单模态聚类基线"
OUT, FIG = HERE / "outputs", HERE / "figures"


def load_by_path(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CB = load_by_path(E1 / "02_cluster_baseline.py", "e1_cluster")
V = load_by_path(E1 / "03_visualize.py", "e1_visualize")
V.OUT, V.FIG = OUT, FIG
F01 = load_by_path(HERE / "01_features.py", "e2_features")
E2C = load_by_path(HERE / "02_cluster_baseline.py", "e2_cluster")
plt, Patch = V.plt, V.Patch
STAGE_C, STAGE_ZH, INK, MUTED, GRID = V.STAGE_C, V.STAGE_ZH, V.INK, V.MUTED, V.GRID

INC = F01.GROUPS["增量场形态"]
TAG = "增量场6"
METHODS = ("KMeans", "GMM", "Ward")
SEEDS = range(20)
MAP_TIMES = [150, 250, 300, 345, 380, 440]
FEAT_ZH = {"inc_P2": "Δγ ≥ 2×中位数 的面积占比", "inc_P4": "Δγ ≥ 4×中位数 的面积占比",
           "inc_q98": "log2(Δγ 的 P98 / P50)", "inc_top5": "最高 5% 测点的 Δγ 份额",
           "inc_nu": "增量泊松比 −med(Δεxx)/med(Δεyy)", "inc_corr": "corr(log Δγ, log γ)"}


def noise_reference(n=400_000, seed=0):
    """空间不相关、各向同性的高斯噪声增量场：Δγ 近似服从 Rayleigh 分布，相对分级特征的理论值。"""
    rng = np.random.default_rng(seed)
    x = np.hypot(rng.normal(size=n), rng.normal(size=n))
    r = x / np.median(x)
    srt = np.sort(x)
    return dict(inc_P2=float((r >= 2).mean()), inc_P4=float((r >= 4).mean()),
                inc_q98=float(np.log2(np.percentile(r, 98))), inc_top5=float(srt[-n // 20:].sum() / srt.sum()),
                analytic_P2="2^-4 = 0.0625", analytic_P4="2^-16 ≈ 1.5e-5", analytic_q98="log2(sqrt(log2 50)) ≈ 1.25")


# ------------------------------------------------------------------ 计算
def run():
    s = pd.read_csv(OUT / "samples.csv")
    y = s.stage_code.to_numpy()
    mask = (y >= 0) & (s.uncertain.to_numpy() == 0)
    sec = s.sec.to_numpy()
    Xs, E, pca, info = F01.E1F.preprocess(s[INC])
    emb = pd.DataFrame(E, columns=[f"PC{i + 1}" for i in range(E.shape[1])])
    emb.insert(0, "sec", s.sec)
    emb.to_csv(OUT / f"embed_DIC_{TAG}.csv", index=False, encoding="utf-8-sig")
    Xs.insert(0, "sec", s.sec)
    Xs.to_csv(OUT / f"scaled_DIC_{TAG}.csv", index=False, encoding="utf-8-sig")

    null = E2C.contiguous_null(len(s), np.where(mask)[0], y)
    pct = lambda a: float((null < a).mean() * 100)
    sil_true = silhouette_score(E[mask], y[mask])
    rows, stab, labels = [], [], {"sec": s.sec, "stage_code": y, "uncertain": s.uncertain}
    for meth in METHODS:
        c = CB.cluster(E, meth)
        labels[f"DIC_{TAG}_{meth}"] = CB.align_to_stages(np.where(mask, y, -1), c)
        r = CB.evaluate(y, c, E, mask)
        off = E2C.boundary_offset(sec, c)
        rows.append(dict(modality="DIC", feature_set=TAG, method=meth, n_pcs=E.shape[1], **r,
                         null_percentile=pct(r["ARI"]), off_ci_s=off[0], off_cd_s=off[1], off_peak_s=off[2],
                         silhouette_true_stage=sil_true,
                         switch_times=str((sec[np.where(np.diff(c) != 0)[0] + 1]).tolist())))
        if meth != "Ward":
            aris = [adjusted_rand_score(y[mask], CB.cluster(E, meth, seed=sd)[mask]) for sd in SEEDS]
            stab.append(dict(method=meth, ARI_mean=np.mean(aris), ARI_std=np.std(aris),
                             ARI_min=np.min(aris), ARI_max=np.max(aris), null_pct_of_mean=pct(np.mean(aris))))
        ct = pd.crosstab(pd.Series(y[mask], name="人工阶段").map(dict(enumerate(STAGE_ZH))),
                         pd.Series(labels[f"DIC_{TAG}_{meth}"][mask], name="簇（已对齐）")).reindex(STAGE_ZH)
        ct.to_csv(OUT / f"inc6_contingency_{meth}.csv", encoding="utf-8-sig")
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "inc6_metrics.csv", index=False, encoding="utf-8-sig")
    st = pd.DataFrame(stab)
    st.to_csv(OUT / "inc6_stability.csv", index=False, encoding="utf-8-sig")
    lab = pd.DataFrame(labels)
    lab.to_csv(OUT / "inc6_cluster_labels.csv", index=False, encoding="utf-8-sig")

    med = []
    for f in INC:
        g = [s.loc[mask & (y == k), f].dropna() for k in range(4)]
        H = kruskal(*g).statistic
        a = s[[f, "sec"]].dropna()
        med.append(dict(feature=f, meaning=FEAT_ZH[f], **{STAGE_ZH[k]: float(g[k].median()) for k in range(4)},
                        eps2=float((H - 3) / (mask.sum() - 4)), rho_time=float(spearmanr(a[f], a.sec)[0])))
    med = pd.DataFrame(med)
    med.to_csv(OUT / "inc6_stage_medians.csv", index=False, encoding="utf-8-sig")
    nref = noise_reference()
    (OUT / "inc6_noise_reference.json").write_text(json.dumps(nref, ensure_ascii=False, indent=1), encoding="utf-8")

    load = pd.DataFrame(pca.components_.T, index=INC, columns=[f"PC{i + 1}" for i in range(E.shape[1])])
    load.to_csv(OUT / "inc6_pca_loadings.csv", encoding="utf-8-sig")
    # 聚类空间按 90% 方差只保留了 1 个主成分；作图另取 2 个主成分（PC1 与聚类空间一致，PC2 只用于展示）
    from sklearn.decomposition import PCA
    Z = Xs.drop(columns="sec").to_numpy(float)
    pca2 = PCA(2, svd_solver="full").fit(Z)
    E2 = pca2.transform(Z)
    if np.corrcoef(E2[:, 0], E[:, 0])[0, 1] < 0:
        E2[:, 0] *= -1
        pca2.components_[0] *= -1
    pd.DataFrame(pca2.components_.T, index=INC, columns=["PC1", "PC2"]).to_csv(
        OUT / "inc6_pca_loadings_2d_display.csv", encoding="utf-8-sig")
    return s, E, E2, pca, pca2, m, st, lab, med, nref, null, sil_true


# ------------------------------------------------------------------ 图
def fig10(s, E1d, E, pca2, m, sil_true):
    """E1d：聚类用的 1 维空间；E：作图用 2 维 PCA（PC1 与 E1d 同向）。"""
    y = s.stage_code.to_numpy()
    unc = s.uncertain.to_numpy() == 1
    cols = [V.UNC_C if unc[i] else V.label_color(y[i]) for i in range(len(s))]
    order = np.r_[np.where(unc)[0], np.where(~unc)[0]]
    var = [float(np.var(E1d[(y == k) & ~unc, 0])) for k in range(4)]
    ev = pca2.explained_variance_ratio_
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
    V.scatter(axs[0], E, cols, f"(a) 按人工阶段着色\n人工阶段轮廓系数 {sil_true:.2f}；各阶段 PC1 方差 "
              + "/".join(f"{v:.1f}" for v in var), order)
    tc = plt.cm.get_cmap(V.SEQ)((s.sec - s.sec.min()) / (s.sec.max() - s.sec.min()))
    axs[1].plot(E[:, 0], E[:, 1], color=MUTED, lw=0.3, alpha=0.6)
    axs[1].scatter(E[:, 0], E[:, 1], c=tc, s=16, lw=0.6, edgecolor="white")
    for t in (100, 270, 334, 362, 427):
        i = int(np.argmin(np.abs(s.sec.to_numpy() - t)))
        axs[1].annotate(f"{t} s", E[i, :2], xytext=(6, 6), textcoords="offset points", fontsize=7, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    axs[1].set_title("(b) 按时间着色（浅→深 = 0→457 s），细线连接相邻秒", fontsize=9)
    ax = axs[2]
    L = pca2.components_[:2].T
    ys = np.arange(len(INC))[::-1]
    w = 0.38
    ax.barh(ys + w / 2, L[:, 0], height=w, color="#2a78d6", label="PC1（聚类空间，94%）")
    ax.barh(ys - w / 2, L[:, 1], height=w, color="#c3c2b7", label="PC2（仅展示，4%）")
    for y_, (a1, a2) in zip(ys, L):
        ax.text(a1 + (0.02 if a1 >= 0 else -0.02), y_ + w / 2, f"{a1:.2f}", va="center",
                ha="left" if a1 >= 0 else "right", fontsize=7)
        ax.text(a2 + (0.02 if a2 >= 0 else -0.02), y_ - w / 2, f"{a2:.2f}", va="center",
                ha="left" if a2 >= 0 else "right", fontsize=7)
    ax.set_yticks(ys, INC, fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.45, 1.1)
    ax.set_xlabel("载荷（标准化特征 → 主成分的权重）", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("(c) 主成分载荷", fontsize=9)
    V.clean(ax, grid_y=False)
    for a in axs[1:2]:
        a.set_xlabel("PC1", fontsize=8)
        a.set_ylabel("PC2", fontsize=8)
        V.clean(a, grid_y=False)
        a.grid(color=GRID, lw=0.5)
    fig.suptitle(f"图 10  仅用 DIC 增量场 6 个特征：预处理后的样本点（每点 = 1 s）\n"
                 f"聚类空间按 90% 方差只保留 PC1（{ev[0]:.0%}）；PC2（{ev[1]:.0%}，主要来自 inc_nu）仅用于展示",
                 fontsize=10.5)
    V.stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(FIG / "fig10_inc6_embedding_stage.png", dpi=120)
    plt.close(fig)


def fig11(s, E, m, lab):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.8), gridspec_kw=dict(height_ratios=[1.25, 1]))
    for j, meth in enumerate(METHODS):
        r = m[m.method == meth].iloc[0]
        lb = lab[f"DIC_{TAG}_{meth}"].to_numpy()
        V.scatter(axs[0, j], E, [V.label_color(v) for v in lb],
                  f"{meth}\nARI = {r.ARI:.2f}   轮廓系数 = {r.silhouette:.2f}   零模型百分位 {r.null_percentile:.0f}%")
        ct = pd.read_csv(OUT / f"inc6_contingency_{meth}.csv", index_col=0)
        ct.columns = [f"簇≈{STAGE_ZH[int(c)]}" if int(c) < 4 else f"额外簇{c}" for c in ct.columns]
        P = ct.div(ct.sum(axis=1), axis=0)
        ax = axs[1, j]
        ax.imshow(P.to_numpy(), cmap=V.SEQ, vmin=0, vmax=1, aspect="auto")
        for i in range(P.shape[0]):
            for k in range(P.shape[1]):
                v = P.iat[i, k]
                ax.text(k, i, f"{v:.0%}\n({ct.iat[i, k]})", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(P.shape[1]), P.columns, fontsize=7, rotation=20)
        ax.set_yticks(range(P.shape[0]), P.index, fontsize=8)
        ax.set_title(f"{meth}：人工阶段（行）在各簇中的分布", fontsize=9)
        V.clean(ax, grid_y=False)
    fig.suptitle("图 11  仅用 DIC 增量场 6 个特征的聚类结果（k = 4；簇颜色已对齐到最接近的人工阶段）", fontsize=11)
    fig.legend([Patch(color=c) for c in STAGE_C], [f"簇 ≈ {t}" for t in STAGE_ZH], frameon=False, fontsize=8,
               ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIG / "fig11_inc6_embedding_cluster.png", dpi=120)
    plt.close(fig)


def fig12(s, m, lab):
    base = pd.read_csv(OUT / "cluster_labels.csv")
    m2 = pd.read_csv(OUT / "metrics_main.csv")
    rows = [(base["时间_基线_KMeans"].to_numpy(), "仅时间 · KMeans",
             m2[(m2.modality == "时间")].iloc[0]),
            (base["DIC_全部_GMM"].to_numpy(), "DIC-SF 全部20 · GMM",
             m2[(m2.modality == "DIC") & (m2.feature_set == "全部") & (m2.method == "GMM")].iloc[0])]
    for meth in METHODS:
        rows.append((lab[f"DIC_{TAG}_{meth}"].to_numpy(), f"增量场6 · {meth}", m[m.method == meth].iloc[0]))
    fig, axs = plt.subplots(len(rows) + 2, 1, figsize=(12, 6.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.2, 0.8] + [0.8] * len(rows)))
    axs[0].plot(s.sec, s.load_kN, color=INK, lw=1.2)
    axs[0].set_ylabel("载荷 (kN)", fontsize=8)
    V.boundaries(axs[0], s)
    V.clean(axs[0])
    V.stage_bar(axs[1], s)
    for ax, (lb, nm, r) in zip(axs[2:], rows):
        for v in sorted(set(lb)):
            mm = lb == v
            ax.bar(s.sec[mm], 1, width=1.0, color=V.label_color(v), lw=0)
        ax.set_yticks([])
        ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
        ax.text(1.005, 0.5, f"ARI {r.ARI:.2f}\n切换 {int(r.n_switches)}\n零模型 {r.null_percentile:.0f}%",
                transform=ax.transAxes, fontsize=7, va="center", color=INK)
        V.boundaries(ax, s)
        V.clean(ax, grid_y=False)
        ax.spines["left"].set_visible(False)
    axs[-1].set_xlabel("时间 (s)", fontsize=9)
    fig.suptitle("图 12  聚类标签沿时间展开：仅用增量场 6 个特征 vs DIC-SF 全部 20 个特征 vs 仅时间", fontsize=11)
    V.stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 0.94, 0.95))
    fig.savefig(FIG / "fig12_inc6_cluster_timeline.png", dpi=120)
    plt.close(fig)


def fig13(s, nref):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    C, MF = F01.C, F01.MF
    a = C.open_cache()
    fi = pd.read_csv(C.OUT / "frame_index.csv")
    XX, YY = C.grid_xy_mm()
    ext = [XX.min(), XX.max(), YY.min(), YY.max()]
    edges = [0, 0.5, 1, 2, 4, 8, 1e9]
    cmap = ListedColormap(["#f4f3f0", "#cde2fb", "#86b6ef", "#2a78d6", "#eb6834", "#b8302f"])
    norm = BoundaryNorm(edges, cmap.N)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.55, 1, 1], hspace=0.55, wspace=0.35)
    for i, t in enumerate(MAP_TIMES):
        k = int(np.argmin(np.abs(fi.t_s.to_numpy() - t)))
        f1, ok1, _ = F01.frame_fields(a, k)
        f0, ok0, _ = F01.frame_fields(a, k - F01.LAG)
        both = ok1 & ok0
        d = {c: f1[c] - f0[c] for c in ("exx", "eyy", "exy")}
        dg = MF.gamma_max(d["exx"], d["eyy"], d["exy"])
        med = np.median(dg[both])
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.where(both, dg / med, np.nan), cmap=cmap, norm=norm, extent=ext, origin="upper",
                  interpolation="nearest")
        stg = STAGE_ZH[int(C.label_times(np.array([t]))[0][0])]
        ax.set_title(f"t = {t} s（{stg}）\nΔγ 中位数 {med * 1e3:.3f}‰ / 5 s", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend([Patch(color=cmap(i)) for i in range(cmap.N)], ["< 0.5×", "0.5–1×", "1–2×", "2–4×", "4–8×", "≥ 8×"],
               title="Δγ（过去 5 s 的剪应变增量）/ 本帧 Δγ 中位数", frameon=False, fontsize=8, title_fontsize=8,
               ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    ref = {"inc_P2": (nref["inc_P2"], "纯噪声理论值"), "inc_P4": (nref["inc_P4"], "纯噪声理论值"),
           "inc_q98": (nref["inc_q98"], "纯噪声理论值"), "inc_top5": (nref["inc_top5"], "纯噪声理论值"),
           "inc_nu": (0.5, "0.5：扩容阈值"), "inc_corr": (0.0, "0：与累积场无关")}
    bnd = [0, 270, 334, 427.3, s.sec.max()]
    for j, f in enumerate(INC):
        ax = fig.add_subplot(gs[1 + j // 3, (j % 3) * 2:(j % 3) * 2 + 2])
        for k in range(4):
            ax.axvspan(bnd[k], bnd[k + 1], color=STAGE_C[k], alpha=0.10, lw=0)
        ax.plot(s.sec, s[f], color=INK, lw=1.1)
        v, txt = ref[f]
        ax.axhline(v, color=MUTED, ls="--", lw=0.9)
        ax.text(s.sec.max(), v, f" {txt}", fontsize=7, color=MUTED, va="bottom", ha="right")
        for t in MAP_TIMES:
            ax.axvline(t, color=MUTED, ls=":", lw=0.6)
        ax.set_title(f"{f}：{FEAT_ZH[f]}", fontsize=9)
        V.clean(ax)
        if j >= 3:
            ax.set_xlabel("时间 (s)", fontsize=8)
    fig.legend([Patch(color=c, alpha=0.3) for c in STAGE_C], STAGE_ZH, frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("图 13  增量场 6 个特征的构造：上 = 过去 5 s 剪应变增量的相对分级图；下 = 6 个特征（逐秒，过去 5 s 平均；"
                 "点线 = 上排时刻）", fontsize=10.5)
    fig.savefig(FIG / "fig13_inc6_feature_construction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    s, E, E2, pca, pca2, m, st, lab, med, nref, null, sil_true = run()
    print(f"增量场 6 特征 → {E.shape[1]} 个主成分，方差占比 {np.round(pca.explained_variance_ratio_, 3).tolist()}")
    print("聚类空间 PC1 载荷 / 展示用 2 维 PCA 载荷（方差占比 %s）：" % np.round(pca2.explained_variance_ratio_, 3).tolist())
    print(pd.concat([pd.read_csv(OUT / "inc6_pca_loadings.csv", index_col=0),
                     pd.read_csv(OUT / "inc6_pca_loadings_2d_display.csv", index_col=0).add_suffix("_展示")], axis=1)
          .round(3).to_string())
    print(f"人工阶段轮廓系数 {sil_true:.3f}")
    print("\n== 聚类（k=4） ==")
    print(m.round(3).to_string(index=False))
    print("\n== 稳定性（20 个种子） ==")
    print(st.round(3).to_string(index=False))
    for meth in METHODS:
        print(f"\n[{meth}] 列联表\n", pd.read_csv(OUT / f"inc6_contingency_{meth}.csv", index_col=0).to_string())
    print("\n== 各特征按阶段中位数、效应量 ε²、与时间 ρ ==")
    print(med.round(3).to_string(index=False))
    print("\n纯噪声参照：", nref)
    fig10(s, E, E2, pca2, m, sil_true)
    fig11(s, E2, m, lab)
    fig12(s, m, lab)
    fig13(s, nref)
    print("figures:", sorted(p.name for p in FIG.glob("fig1[0-3]*.png")))


if __name__ == "__main__":
    main()
