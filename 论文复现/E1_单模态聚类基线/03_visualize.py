"""03 可视化：预处理样本、所提特征、聚类效果。

图
  fig1_features_timeline.png   各特征（鲁棒标准化后）随时间的热图 + 载荷与阶段条，AE / DIC 上下对照
  fig2_feature_by_stage.png    每个特征按人工阶段的分布（箱线），AE / DIC 各一页
  fig3_embedding_stage.png     预处理后 PCA 前两维散点：按人工阶段着色（AE vs DIC）
  fig4_embedding_cluster.png   同一散点按聚类结果着色（KMeans / GMM / Ward），标注 ARI 与轮廓系数
  fig5_cluster_timeline.png    聚类标签沿时间展开，与人工阶段、载荷对照
  fig6_metrics.png             主指标对比（ARI 为主，轮廓系数为辅）+ k 扫描
  fig7_contingency.png         聚类–阶段列联表（行归一化）
  fig8_silhouette.png          各样本轮廓值（按簇排序），AE / DIC 最佳方法
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402
from sklearn.metrics import silhouette_samples        # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STAGE_ZH = ["起裂前", "起裂至损伤", "损伤至峰值", "峰后"]
STAGE_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # 分类色 1–4（固定顺序）
EXTRA_C = ["#e87ba4", "#008300", "#6250d6", "#e34948"]       # k>4 时多余簇
UNC_C = "#c3c2b7"
MOD_C = {"AE": "#2a78d6", "DIC": "#eb6834", "时间": "#8a8984"}
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19b8f", "#b8302f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3f0", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])


def clean(ax, grid_y=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def label_color(v):
    v = int(v)
    if v < 0:
        return UNC_C
    if v < 4:
        return STAGE_C[v]
    return EXTRA_C[(v - 10) % 4]


def load():
    s = pd.read_csv(OUT / "samples.csv")
    cat = pd.read_csv(OUT / "feature_catalog.csv")
    lab = pd.read_csv(OUT / "cluster_labels.csv")
    m = pd.read_csv(OUT / "metrics_main.csv")
    ks = pd.read_csv(OUT / "metrics_k_scan.csv")
    sc = {mod: pd.read_csv(OUT / f"scaled_{mod}.csv") for mod in ("AE", "DIC")}
    emb = {(mod, tag): pd.read_csv(OUT / f"embed_{mod}_{tag}.csv") for mod in ("AE", "DIC") for tag in ("全部", "去趋势")}
    return s, cat, lab, m, ks, sc, emb


def stage_bar(ax, s, y=None, label="人工阶段"):
    y = s.stage_code.to_numpy() if y is None else y
    unc = s.uncertain.to_numpy() == 1
    for v in sorted(set(y)):
        mm = (y == v) & ~unc
        ax.bar(s.sec[mm], 1, width=1.0, color=label_color(v), lw=0)
    ax.bar(s.sec[unc], 1, width=1.0, color=UNC_C, lw=0)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    clean(ax, grid_y=False)
    ax.spines["left"].set_visible(False)


def boundaries(ax, s):
    y = s.stage_code.to_numpy()
    for i in np.where(np.diff(y) != 0)[0]:
        if y[i] >= 0 and y[i + 1] >= 0:
            ax.axvline(s.sec[i] + 0.5, color=INK, lw=0.8, ls="--", alpha=0.7)


def stage_legend(fig, extra=None, y=0.0):
    h = [Patch(color=c) for c in STAGE_C] + [Patch(color=UNC_C)]
    t = STAGE_ZH + ["不确定（边界 ±1 s / 峰值区间）"]
    if extra:
        h += extra[0]
        t += extra[1]
    fig.legend(h, t, frameon=False, fontsize=8, ncol=len(t), loc="lower center", bbox_to_anchor=(0.5, y))


# ------------------------------------------------------------------ fig1
def fig_features_timeline(s, cat, sc):
    fig = plt.figure(figsize=(12, 11))
    n_ae, n_dic = (cat.modality == "AE").sum(), (cat.modality == "DIC").sum()
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 0.35, n_ae, n_dic], width_ratios=[40, 1], hspace=0.08, wspace=0.02)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(s.sec, s.load_kN, color=INK, lw=1.2)
    ax0.set_ylabel("载荷 (kN)", fontsize=9)
    boundaries(ax0, s)
    clean(ax0)
    ax0.set_xticklabels([])
    axb = fig.add_subplot(gs[1, 0], sharex=ax0)
    stage_bar(axb, s)
    axb.set_xticklabels([])
    ims = None
    for row, mod in ((2, "AE"), (3, "DIC")):
        ax = fig.add_subplot(gs[row, 0], sharex=ax0)
        c = cat[cat.modality == mod].sort_values(["kind", "feature"])
        M = sc[mod][c.feature].to_numpy().T
        ims = ax.imshow(M, aspect="auto", cmap=DIV, vmin=-4, vmax=4, interpolation="nearest",
                        extent=[s.sec.min() - 0.5, s.sec.max() + 0.5, len(c) - 0.5, -0.5])
        ax.set_yticks(range(len(c)), [f"{f}  ·{g}" for f, g in zip(c.feature, c.group_zh)], fontsize=7)
        ax.set_ylabel(f"{mod} 特征", fontsize=10, color=MOD_C[mod])
        boundaries(ax, s)
        clean(ax, grid_y=False)
        if mod == "AE":
            ax.set_xticklabels([])
    ax.set_xlabel("时间 (s)", fontsize=9)
    cax = fig.add_subplot(gs[2:, 1])
    cb = fig.colorbar(ims, cax=cax)
    cb.set_label("鲁棒标准化值（中位数=0，IQR 尺度，截尾 ±5）", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("图 1  预处理后的逐秒特征（虚线：人工阶段边界）", fontsize=11, x=0.45)
    stage_legend(fig, y=0.005)
    fig.subplots_adjust(left=0.2, right=0.93, top=0.95, bottom=0.07)
    fig.savefig(FIG / "fig1_features_timeline.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ fig2
def fig_feature_by_stage(s, cat):
    mask = (s.stage_code >= 0) & (s.uncertain == 0)
    for mod in ("AE", "DIC"):
        c = cat[cat.modality == mod].sort_values(["kind", "feature"])
        n = len(c)
        ncol = 5
        nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.4 * nrow))
        axs = axs.ravel()
        for ax, r in zip(axs, c.itertuples()):
            data = [s.loc[mask & (s.stage_code == k), r.feature].dropna().to_numpy() for k in range(4)]
            bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED, lw=0.8),
                            capprops=dict(color=MUTED, lw=0.8))
            for patch, col in zip(bp["boxes"], STAGE_C):
                patch.set(facecolor=col, edgecolor="white", alpha=0.85)
            ax.set_title(f"{r.feature}\n[{r.group_zh}]", fontsize=8)
            ax.set_xticks(range(1, 5), ["前", "裂–损", "损–峰", "后"], fontsize=7)
            clean(ax)
        for ax in axs[n:]:
            ax.axis("off")
        fig.suptitle(f"图 2{'a' if mod == 'AE' else 'b'}  {mod} 特征按人工阶段的分布（原始单位，箱线不含离群点）",
                     fontsize=11)
        h = [Patch(color=cc) for cc in STAGE_C]
        fig.legend(h, STAGE_ZH, frameon=False, fontsize=8, ncol=4, loc="lower center")
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(FIG / f"fig2{'a' if mod == 'AE' else 'b'}_{mod}_feature_by_stage.png", dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------ fig3 / fig4
def scatter(ax, E, colors, title, order=None):
    idx = np.arange(len(E)) if order is None else order
    ax.scatter(E[idx, 0], E[idx, 1], c=[colors[i] for i in idx], s=16, lw=0.6, edgecolor="white", alpha=0.95)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    clean(ax, grid_y=False)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def fig_embedding_stage(s, emb, m):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.6))
    y = s.stage_code.to_numpy()
    unc = s.uncertain.to_numpy() == 1
    cols = [UNC_C if unc[i] else label_color(y[i]) for i in range(len(s))]
    order = np.r_[np.where(unc)[0], np.where(~unc)[0]]
    for r, mod in enumerate(("AE", "DIC")):
        E = emb[(mod, "全部")].filter(like="PC").to_numpy()
        sil = m[(m.modality == mod) & (m.feature_set == "全部")].silhouette_true_stage.iloc[0]
        scatter(axs[r, 0], E, cols, f"{mod}：PCA 前两维，按人工阶段着色\n人工阶段轮廓系数 = {sil:.2f}", order)
        # 时间着色
        tc = plt.cm.get_cmap(SEQ)((s.sec - s.sec.min()) / (s.sec.max() - s.sec.min()))
        axs[r, 1].scatter(E[:, 0], E[:, 1], c=tc, s=16, lw=0.6, edgecolor="white")
        axs[r, 1].plot(E[:, 0], E[:, 1], color=MUTED, lw=0.3, alpha=0.5)
        axs[r, 1].set_title(f"{mod}：按时间着色（浅→深 = 0→457 s），细线连接相邻秒", fontsize=9)
        axs[r, 1].set_xlabel("PC1", fontsize=8)
        axs[r, 1].set_ylabel("PC2", fontsize=8)
        clean(axs[r, 1], grid_y=False)
        axs[r, 1].grid(color=GRID, lw=0.5)
        # 去趋势
        E2 = emb[(mod, "去趋势")].filter(like="PC").to_numpy()
        sil2 = m[(m.modality == mod) & (m.feature_set == "去趋势")].silhouette_true_stage.iloc[0]
        scatter(axs[r, 2], E2, cols, f"{mod}（去趋势特征集）：按人工阶段着色\n人工阶段轮廓系数 = {sil2:.2f}", order)
    fig.suptitle("图 3  预处理后的样本点（每点 = 1 s；PCA 保留 90% 方差，此处只画前两维）", fontsize=11)
    stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIG / "fig3_embedding_stage.png", dpi=120)
    plt.close(fig)


def fig_embedding_cluster(s, emb, lab, m):
    methods = ["KMeans", "GMM", "Ward"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.6))
    for r, mod in enumerate(("AE", "DIC")):
        E = emb[(mod, "全部")].filter(like="PC").to_numpy()
        for c_i, meth in enumerate(methods):
            lb = lab[f"{mod}_全部_{meth}"].to_numpy()
            mm = m[(m.modality == mod) & (m.feature_set == "全部") & (m.method == meth)].iloc[0]
            cols = [label_color(v) for v in lb]
            scatter(axs[r, c_i], E, cols, f"{mod} · {meth}\nARI = {mm.ARI:.2f}   轮廓系数 = {mm.silhouette:.2f}")
    fig.suptitle("图 4  聚类结果（k = 4；簇颜色已用匈牙利匹配对齐到最接近的人工阶段，便于与图 3 对照）", fontsize=11)
    h = [Patch(color=c) for c in STAGE_C]
    fig.legend(h, [f"簇 ≈ {t}" for t in STAGE_ZH], frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIG / "fig4_embedding_cluster.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ fig5
def fig_cluster_timeline(s, lab, m):
    rows = [("时间_基线_KMeans", "仅时间 · KMeans")]
    for mod in ("AE", "DIC"):
        for meth in ("KMeans", "GMM", "Ward"):
            rows.append((f"{mod}_全部_{meth}", f"{mod} · {meth}"))
    fig, axs = plt.subplots(len(rows) + 2, 1, figsize=(12, 7.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.2, 0.8] + [0.8] * len(rows)))
    axs[0].plot(s.sec, s.load_kN, color=INK, lw=1.2)
    axs[0].set_ylabel("载荷 (kN)", fontsize=8)
    boundaries(axs[0], s)
    clean(axs[0])
    stage_bar(axs[1], s)
    for ax, (col, nm) in zip(axs[2:], rows):
        lb = lab[col].to_numpy()
        for v in sorted(set(lb)):
            mm = lb == v
            ax.bar(s.sec[mm], 1, width=1.0, color=label_color(v), lw=0)
        key = col.split("_")
        mrow = m[(m.modality == key[0]) & (m.feature_set == key[1]) & (m.method == key[2])].iloc[0]
        ax.set_yticks([])
        ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
        ax.text(1.005, 0.5, f"ARI {mrow.ARI:.2f}\n切换 {mrow.n_switches}", transform=ax.transAxes,
                fontsize=7, va="center", color=INK)
        boundaries(ax, s)
        clean(ax, grid_y=False)
        ax.spines["left"].set_visible(False)
    axs[-1].set_xlabel("时间 (s)", fontsize=9)
    fig.suptitle("图 5  聚类标签沿时间展开（颜色 = 对齐后的阶段；人工阶段只有 3 次切换）", fontsize=11)
    stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 0.95, 0.96))
    fig.savefig(FIG / "fig5_cluster_timeline.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ fig6
def fig_metrics(m, ks):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.4))
    mm = m[m.method.isin(["KMeans", "GMM", "Ward"])].copy()
    mm["name"] = mm.modality + "·" + mm.feature_set
    order = ["AE·全部", "AE·去趋势", "DIC·全部", "DIC·去趋势", "时间·基线"]
    meth = ["KMeans", "GMM", "Ward"]
    MC = ["#2a78d6", "#eb6834", "#1baf7a"]
    for ax, metric, title in ((axs[0, 0], "ARI", "ARI（主指标：与人工阶段一致度）"),
                              (axs[0, 1], "silhouette", "轮廓系数（辅指标：簇自身紧致/分离度）")):
        x = np.arange(len(order))
        for i, me in enumerate(meth):
            v = [mm[(mm.name == o) & (mm.method == me)][metric].mean() if ((mm.name == o) & (mm.method == me)).any()
                 else np.nan for o in order]
            bars = ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=MC[i], label=me)
            for b, val in zip(bars, v):
                if np.isfinite(val):
                    ax.text(b.get_x() + b.get_width() / 2, val + (0.01 if val >= 0 else -0.04), f"{val:.2f}",
                            ha="center", fontsize=7, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, order, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.1, 1.0)
        clean(ax)
    axs[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    for ax, metric, title in ((axs[1, 0], "ARI", "ARI 随簇数 k（各 k 取三种方法中的最大值）"),
                              (axs[1, 1], "silhouette", "轮廓系数随簇数 k（KMeans）")):
        LC = {"AE·全部": "#2a78d6", "AE·去趋势": "#86b6ef", "DIC·全部": "#eb6834", "DIC·去趋势": "#f2a27f"}
        for nm, col in LC.items():
            mod, tag = nm.split("·")
            d = ks[(ks.modality == mod) & (ks.feature_set == tag)]
            d = d.groupby("k")[metric].max() if metric == "ARI" else d[d.method == "KMeans"].set_index("k")[metric]
            ax.plot(d.index, d.values, color=col, lw=2, marker="o", ms=5, label=nm,
                    ls="-" if tag == "全部" else "--")
        ax.axvline(4, color=MUTED, lw=0.8, ls=":")
        ax.text(4.05, 0.95, "k = 4（人工阶段数）", fontsize=7, color=MUTED, va="top")
        ax.set_xlabel("k", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.1, 1.0)
        clean(ax)
    axs[1, 0].legend(frameon=False, fontsize=8, ncol=2, loc="center right")
    fig.suptitle("图 6  单模态聚类基线的阶段可分性", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig6_metrics.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ fig7
def fig_contingency():
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.8))
    for r, mod in enumerate(("AE", "DIC")):
        for c_i, meth in enumerate(("KMeans", "GMM", "Ward")):
            ct = pd.read_csv(OUT / f"contingency_{mod}_{meth}.csv", index_col=0)
            ct.columns = [f"簇≈{STAGE_ZH[int(c)]}" if int(c) < 4 else f"额外簇{c}" for c in ct.columns]
            P = ct.div(ct.sum(axis=1), axis=0)
            ax = axs[r, c_i]
            ax.imshow(P.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P.iat[i, j]
                    ax.text(j, i, f"{v:.0%}\n({ct.iat[i, j]})", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.55 else INK)
            ax.set_xticks(range(P.shape[1]), P.columns, fontsize=7, rotation=20)
            ax.set_yticks(range(P.shape[0]), P.index, fontsize=8)
            ax.set_title(f"{mod} · {meth}", fontsize=9)
            clean(ax, grid_y=False)
    fig.suptitle("图 7  人工阶段（行）在各簇（列）中的分布：行百分比（括号内为秒数）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig7_contingency.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ fig8
def fig_silhouette(s, emb, lab, m):
    mask = ((s.stage_code >= 0) & (s.uncertain == 0)).to_numpy()
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.6), sharey=False)
    panels = [("AE", "KMeans", "簇"), ("AE", None, "人工阶段"), ("DIC", "GMM", "簇"), ("DIC", None, "人工阶段")]
    for ax, (mod, meth, what) in zip(axs, panels):
        E = emb[(mod, "全部")].filter(like="PC").to_numpy()[mask]
        lb = (lab[f"{mod}_全部_{meth}"] if meth else s.stage_code).to_numpy()[mask]
        sv = silhouette_samples(E, lb)
        y0 = 0
        for v in sorted(set(lb)):
            vals = np.sort(sv[lb == v])
            ax.barh(np.arange(y0, y0 + len(vals)), vals, height=1.0, color=label_color(v), lw=0)
            y0 += len(vals) + 8
        ax.axvline(sv.mean(), color=INK, lw=1, ls="--")
        ax.text(sv.mean(), y0, f" 平均 {sv.mean():.2f}", fontsize=8, va="bottom")
        ax.set_yticks([])
        ax.set_xlim(-0.6, 1)
        ax.set_xlabel("样本轮廓值", fontsize=8)
        ax.set_title(f"{mod} · 按{what}" + (f"（{meth}）" if meth else ""), fontsize=9)
        clean(ax, grid_y=False)
    fig.suptitle("图 8  逐样本轮廓值：负值 = 该样本离别的组更近", fontsize=11)
    h = [Patch(color=c) for c in STAGE_C]
    fig.legend(h, [f"阶段 / 簇≈{t}" for t in STAGE_ZH], frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIG / "fig8_silhouette.png", dpi=120)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s, cat, lab, m, ks, sc, emb = load()
    fig_features_timeline(s, cat, sc)
    fig_feature_by_stage(s, cat)
    fig_embedding_stage(s, emb, m)
    fig_embedding_cluster(s, emb, lab, m)
    fig_cluster_timeline(s, lab, m)
    fig_metrics(m, ks)
    fig_contingency()
    fig_silhouette(s, emb, lab, m)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
