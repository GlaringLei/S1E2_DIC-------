"""03 E2 可视化。图号与 E1 对应，便于并排比较；图 3、6 重写，另加图 9（Dong 式相对分级示意）。

  fig1_features_timeline.png   预处理后的逐秒特征热图（复用 E1 函数）
  fig2a/2b_*_feature_by_stage  各特征按阶段箱线（复用 E1）
  fig3_embedding_stage.png     PCA 散点：人工阶段 / 时间着色；E1 vs E2 的 DIC 空间对照
  fig4_embedding_cluster.png   聚类结果散点（复用 E1）
  fig5_cluster_timeline.png    聚类标签沿时间（复用 E1）
  fig6_metrics.png             ARI（主）/ 轮廓系数（辅）、与 E1 对照、零模型线、k 扫描、特征组消融
  fig7_contingency.png         列联表（复用 E1）
  fig8_silhouette.png          逐样本轮廓值（复用 E1）
  fig9_dong_relative_grades.png  相对分级图（SCCM 尺度无关版）在代表帧上的样子 + 占比曲线
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "E1_单模态聚类基线"
OUT, FIG = HERE / "outputs", HERE / "figures"
FIG.mkdir(exist_ok=True)


def load_by_path(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = load_by_path(E1 / "03_visualize.py", "e1_visualize")
V.OUT, V.FIG = OUT, FIG                     # 复用 E1 的作图函数，输入输出改到 E2
F01 = load_by_path(HERE / "01_features.py", "e2_features")
plt = V.plt
Patch = V.Patch
STAGE_C, STAGE_ZH, INK, MUTED, GRID = V.STAGE_C, V.STAGE_ZH, V.INK, V.MUTED, V.GRID


def fig3(s, emb, m):
    e1emb = pd.read_csv(E1 / "outputs" / "embed_DIC_全部.csv").filter(like="PC").to_numpy()
    e1m = pd.read_csv(E1 / "outputs" / "metrics_main.csv")
    y = s.stage_code.to_numpy()
    unc = s.uncertain.to_numpy() == 1
    cols = [V.UNC_C if unc[i] else V.label_color(y[i]) for i in range(len(s))]
    order = np.r_[np.where(unc)[0], np.where(~unc)[0]]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.6))
    panels = [(axs[0, 0], emb[("AE", "全部")], m, "AE", "全部", "AE·因果：按人工阶段"),
              (axs[1, 0], e1emb, e1m, "DIC", "全部", "E1 DIC-Ren（幅值特征）：按人工阶段"),
              (axs[1, 1], emb[("DIC", "全部")], m, "DIC", "全部", "E2 DIC-SF（尺度无关）：按人工阶段"),
              (axs[1, 2], emb[("DIC", "增量动态")], m, "DIC", "增量动态", "E2 DIC-SF 增量动态组：按人工阶段")]
    for ax, e, mm, mod, tag, title in panels:
        E = e if isinstance(e, np.ndarray) else e.filter(like="PC").to_numpy()
        sil = mm[(mm.modality == mod) & (mm.feature_set == tag)].silhouette_true_stage.iloc[0]
        var = [np.trace(np.cov(E[(y == k) & ~unc].T)) for k in range(4)]
        V.scatter(ax, E, cols, f"{title}\n人工阶段轮廓系数 {sil:.2f}；各阶段方差 " + "/".join(f"{v:.1f}" for v in var), order)
    for ax, mod in ((axs[0, 1], "AE"), (axs[0, 2], "DIC")):
        E = emb[(mod, "全部")].filter(like="PC").to_numpy()
        tc = plt.cm.get_cmap(V.SEQ)((s.sec - s.sec.min()) / (s.sec.max() - s.sec.min()))
        ax.plot(E[:, 0], E[:, 1], color=MUTED, lw=0.3, alpha=0.5)
        ax.scatter(E[:, 0], E[:, 1], c=tc, s=16, lw=0.6, edgecolor="white")
        ax.set_title(f"E2 {mod}：按时间着色（浅→深 = 0→457 s）", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        V.clean(ax, grid_y=False)
        ax.grid(color=GRID, lw=0.5)
    fig.suptitle("图 3  预处理后的样本点（每点 = 1 s；PCA 前两维）。下排对比 E1 与 E2 的 DIC 空间", fontsize=11)
    V.stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIG / "fig3_embedding_stage.png", dpi=120)
    plt.close(fig)


def fig6(m, ks, ab):
    e1m = pd.read_csv(E1 / "outputs" / "metrics_main.csv")
    null = pd.read_csv(OUT / "null_model.csv", index_col=0).iloc[:, 0]
    q50, q95 = float(null["50%"]), float(null["95%"])
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.8))
    meth = ["KMeans", "GMM", "Ward"]
    MC = ["#2a78d6", "#eb6834", "#1baf7a"]
    groups = [("E1", "AE", "全部"), ("E2", "AE", "全部"), ("E1", "DIC", "全部"), ("E2", "DIC", "全部"),
              ("E2", "DIC", "增量动态"), ("E2", "DIC", "累积场"), ("—", "时间", "基线")]
    names = ["E1 AE", "E2 AE·因果", "E1 DIC-Ren", "E2 DIC-SF", "E2 SF·增量动态", "E2 SF·累积场", "仅时间"]
    for ax, metric, title in ((axs[0, 0], "ARI", "ARI（主指标）"), (axs[0, 1], "silhouette", "轮廓系数（辅指标）")):
        x = np.arange(len(groups))
        for i, me in enumerate(meth):
            vals = []
            for ex, mod, tag in groups:
                src = e1m if ex == "E1" else m
                r = src[(src.modality == mod) & (src.feature_set == tag) & (src.method == me)]
                vals.append(float(r[metric].iloc[0]) if len(r) else np.nan)
            bars = ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=MC[i], label=me)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.05), f"{v:.2f}",
                            ha="center", fontsize=6.5)
        if metric == "ARI":
            ax.axhline(q50, color=MUTED, ls=":", lw=1)
            ax.axhline(q95, color=MUTED, ls="--", lw=1)
            ax.text(len(groups) - 0.5, q95 + 0.01, "随机连续分段 95% 分位", fontsize=7, color=MUTED, ha="right")
            ax.text(len(groups) - 0.5, q50 + 0.01, "中位数", fontsize=7, color=MUTED, ha="right")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, names, fontsize=7.5, rotation=15)
        ax.set_ylim(-0.1, 1.0)
        ax.set_title(title, fontsize=10)
        V.clean(ax)
    axs[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax = axs[1, 0]
    LC = {("AE", "全部"): ("#2a78d6", "-"), ("DIC", "全部"): ("#eb6834", "-"),
          ("DIC", "增量动态"): ("#eb6834", "--"), ("DIC", "累积场"): ("#f2a27f", ":")}
    for (mod, tag), (col, ls) in LC.items():
        d = ks[(ks.modality == mod) & (ks.feature_set == tag)].groupby("k").ARI.max()
        ax.plot(d.index, d.values, color=col, ls=ls, lw=2, marker="o", ms=5, label=f"{mod}·{tag}")
    ax.axhline(q95, color=MUTED, ls="--", lw=1)
    ax.axvline(4, color=MUTED, ls=":", lw=0.8)
    ax.set_xlabel("k", fontsize=8)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("ARI 随簇数 k（各 k 取三方法最大值）", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    V.clean(ax)
    ax = axs[1, 1]
    g = ab[ab.method == "GMM"].copy()
    g["name"] = g.modality + "·" + g.group
    g = g.sort_values(["modality", "ARI"])
    cols = ["#2a78d6" if mm == "AE" else "#eb6834" for mm in g.modality]
    ax.barh(g.name, g.ARI, color=cols)
    for i, r in enumerate(g.itertuples()):
        ax.text(max(r.ARI, 0) + 0.01, i, f"{r.ARI:.2f}  |ρ_t| {r.median_abs_rho_time:.2f}", va="center", fontsize=7)
    ax.axvline(q95, color=MUTED, ls="--", lw=1)
    ax.set_xlim(0, 1.05)
    ax.set_title("特征组消融（GMM ARI；|ρ_t| = 特征与时间相关的中位数）", fontsize=10)
    V.clean(ax, grid_y=False)
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("图 6  单模态聚类基线的阶段可分性：E2 与 E1 对照", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig6_metrics.png", dpi=120)
    plt.close(fig)


def fig9(s):
    """Dong SCCM 的尺度无关版：以本帧 γ 中位数为单位的相对等级图。"""
    sys.path.insert(0, str(HERE.parent.parent / "methodScripts"))
    import common as C
    MF = F01.MF
    from matplotlib.colors import BoundaryNorm, ListedColormap
    a = C.open_cache()
    fi = pd.read_csv(C.OUT / "frame_index.csv")
    times = [60, 200, 300, 380, 420, 445]
    edges = [0, 0.5, 1, 2, 4, 8, 1e9]
    cmap = ListedColormap(["#f4f3f0", "#cde2fb", "#86b6ef", "#2a78d6", "#eb6834", "#b8302f"])
    norm = BoundaryNorm(edges, cmap.N)
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, len(times), height_ratios=[1.6, 1], hspace=0.35)
    XX, YY = C.grid_xy_mm()
    ext = [XX.min(), XX.max(), YY.min(), YY.max()]
    for i, t in enumerate(times):
        k = int(np.argmin(np.abs(fi.t_s.to_numpy() - t)))
        f, ok, _ = F01.frame_fields(a, k)
        g, _, _ = F01.principal(f)
        r = np.where(ok, g / np.median(g[ok]), np.nan)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(r, cmap=cmap, norm=norm, extent=ext, origin="upper", interpolation="nearest")
        stg = V.STAGE_ZH[int(C.label_times(np.array([t]))[0][0])]
        ax.set_title(f"t = {t} s（{stg}）\nγ 中位数 {np.median(g[ok]) * 1e3:.2f}‰", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    for c, col, nm in (("sf_P2", "#86b6ef", "γ ≥ 2×中位数"), ("sf_P4", "#2a78d6", "γ ≥ 4×中位数"),
                       ("sf_P8", "#b8302f", "γ ≥ 8×中位数"), ("sf_e2low", "#1baf7a", "|ε2| < 0.5×中位数（Dong ε2 初始等级）"),
                       ("sf_Plow", "#8a8984", "γ < 0.5×中位数")):
        ax.plot(s.sec, s[c] * 100, color=col, lw=1.6, label=nm)
    for x in (270, 334, 427.3):
        ax.axvline(x, color=INK, ls="--", lw=0.8)
    for x in times:
        ax.axvline(x, color=MUTED, ls=":", lw=0.6)
    ax.set_ylabel("面积占比 (%)", fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    V.clean(ax)
    h = [Patch(color=cmap(i)) for i in range(cmap.N)]
    fig.legend(h, ["< 0.5×", "0.5–1×", "1–2×", "2–4×", "4–8×", "≥ 8×"], title="γmax / 本帧中位数",
               frameon=False, fontsize=8, title_fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("图 9  Dong SCCM 的尺度无关版：以本帧 γmax 中位数为单位的相对等级（上），及各等级面积占比（下，虚线 = 人工边界）",
                 fontsize=10.5)
    fig.savefig(FIG / "fig9_dong_relative_grades.png", dpi=120)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = pd.read_csv(OUT / "samples.csv")
    cat = pd.read_csv(OUT / "feature_catalog.csv")
    lab = pd.read_csv(OUT / "cluster_labels.csv")
    m = pd.read_csv(OUT / "metrics_main.csv")
    ks = pd.read_csv(OUT / "metrics_k_scan.csv")
    ab = pd.read_csv(OUT / "metrics_group_ablation.csv")
    sc = {mod: pd.read_csv(OUT / f"scaled_{mod}.csv") for mod in ("AE", "DIC")}
    emb = {(mod, tag): pd.read_csv(OUT / f"embed_{mod}_{tag}.csv")
           for mod, tag in [("AE", "全部"), ("AE", "去趋势"), ("DIC", "全部"), ("DIC", "累积场"), ("DIC", "增量动态")]}
    V.fig_features_timeline(s, cat, sc)
    V.fig_feature_by_stage(s, cat)
    fig3(s, emb, m)
    V.fig_embedding_cluster(s, emb, lab, m)
    V.fig_cluster_timeline(s, lab, m)
    fig6(m, ks, ab)
    V.fig_contingency()
    V.fig_silhouette(s, emb, lab, m)
    fig9(s)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
