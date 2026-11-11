"""03 E3 可视化（与 E1/E2 的图号对应：3 样本分布、4/7 聚类与列联表、5 时间展开、6 指标；另有 Dong 专题图）。

  fig3_embedding_transforms.png   代表性变换下的样本分布（按人工阶段着色）：T0 / T1 / T4(2块) / T4(3块) / T5 拼Dong / REF LDA
  fig4_best_cluster.png           最佳无监督变换：聚类散点 + 列联表 + 主成分载荷
  fig5_cluster_timeline.png       聚类标签沿时间展开：各变换对照
  fig6_transform_metrics.png      聚合/离散指标（轮廓系数、Fisher 比、intra/min）与 ARI、零模型
  fig7_dong_sccm.png              Dong SCCM 原文分级在 HS-2 上：32 级占比热图、判据 1 的阈值-时刻曲线、判据 2
  fig8_dong_vs_inc.png            Dong 固定量程特征 vs 增量场特征：与阶段 ε²、与时间 ρ、相邻阶段区分度
"""
from __future__ import annotations

import importlib.util
import json
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
V.OUT, V.FIG = OUT, FIG
TE = load_by_path(HERE / "02_transform_eval.py", "e3_eval")
plt, Patch = V.plt, V.Patch
STAGE_C, STAGE_ZH, INK, MUTED, GRID = V.STAGE_C, V.STAGE_ZH, V.INK, V.MUTED, V.GRID
SHOW = [0, 1, 5, 6, 10, 13]


def short(name):
    return name.split("（")[0] if len(name) > 26 else name


def fig3(s, tm):
    y = s.stage_code.to_numpy()
    unc = s.uncertain.to_numpy() == 1
    cols = np.array([V.UNC_C if unc[i] else V.label_color(y[i]) for i in range(len(s))], dtype=object)
    order = np.r_[np.where(unc)[0], np.where(~unc)[0]]
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 9))
    rng = np.random.default_rng(0)
    for ax, i in zip(axs.ravel(), SHOW):
        r = tm[tm.id == i].iloc[0]
        E = pd.read_csv(OUT / f"embed_{i:02d}.csv").drop(columns="sec").to_numpy()
        if E.shape[1] == 1:
            E = np.c_[E[:, 0], (y + rng.uniform(-0.3, 0.3, len(y))) * -1]
            ylabel = "人工阶段（仅为错开显示）"
        else:
            ylabel = "C2"
        ax.scatter(E[order, 0], E[order, 1], c=list(cols[order]), s=14, lw=0.5, edgecolor="white")
        ax.set_title(f"{r['transform']}\n轮廓系数 {r.silhouette_true:.2f}  Fisher {r.fisher:.1f}  "
                     f"ARI {max(r.ARI_GMM, r.ARI_Ward):.2f}", fontsize=8.5)
        ax.set_xlabel("C1", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        V.clean(ax, grid_y=False)
        ax.grid(color=GRID, lw=0.5)
    fig.suptitle("图 3  不同尺度无关变换下的样本分布（按人工阶段着色；轮廓系数 / Fisher 比衡量阶段内聚合、阶段间离散）",
                 fontsize=11)
    V.stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIG / "fig3_embedding_transforms.png", dpi=120)
    plt.close(fig)


def fig4(s, tm, lab):
    unsup = tm[~tm["transform"].str.startswith("REF")]
    r = unsup.loc[unsup[["ARI_GMM", "ARI_Ward"]].max(axis=1).idxmax()]
    E = pd.read_csv(OUT / f"embed_{int(r['id']):02d}.csv").drop(columns="sec").to_numpy()
    y = s.stage_code.to_numpy()
    unc = s.uncertain.to_numpy() == 1
    lb = lab[r["transform"]].to_numpy()
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
    V.scatter(axs[0], E, [V.UNC_C if unc[i] else V.label_color(y[i]) for i in range(len(s))],
              f"(a) 按人工阶段着色\n轮廓系数 {r.silhouette_true:.2f}")
    V.scatter(axs[1], E, [V.label_color(v) for v in lb],
              f"(b) 按聚类结果着色（{'GMM' if r.ARI_GMM >= r.ARI_Ward else 'Ward'}）\n"
              f"ARI {max(r.ARI_GMM, r.ARI_Ward):.2f}  零模型 {r.null_pct:.1f}%")
    ct = pd.read_csv(OUT / "contingency_best.csv", index_col=0)
    P = ct.div(ct.sum(axis=1), axis=0)
    ax = axs[2]
    ax.imshow(P.to_numpy(), cmap=V.SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(P.shape[0]):
        for k in range(P.shape[1]):
            v = P.iat[i, k]
            ax.text(k, i, f"{v:.0%}\n({ct.iat[i, k]})", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(4), [f"簇≈{t}" for t in STAGE_ZH], fontsize=7, rotation=20)
    ax.set_yticks(range(4), STAGE_ZH, fontsize=8)
    ax.set_title("(c) 列联表（行百分比）", fontsize=9)
    V.clean(ax, grid_y=False)
    # 载荷：重算 MFA 的 PCA 载荷
    from sklearn.decomposition import PCA
    blocks = [TE.A, ["inc_nu"], ["inc_corr"]]
    Zs, names = [], []
    for b in blocks:
        Z = TE.rz(s[b]).to_numpy()
        Z = Z - Z.mean(0)
        Zs.append(Z / np.linalg.svd(Z, compute_uv=False)[0])
        names += b
    p = PCA(2, svd_solver="full").fit(np.hstack(Zs))
    L = p.components_.T
    ax = axs[3]
    yy = np.arange(len(names))[::-1]
    ax.barh(yy + 0.19, L[:, 0], height=0.38, color="#2a78d6", label=f"C1（{p.explained_variance_ratio_[0]:.0%}）")
    ax.barh(yy - 0.19, L[:, 1], height=0.38, color="#eb6834", label=f"C2（{p.explained_variance_ratio_[1]:.0%}）")
    ax.set_yticks(yy, names, fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("(d) MFA 后的主成分载荷", fontsize=9)
    V.clean(ax, grid_y=False)
    fig.suptitle(f"图 4  最佳无监督变换：{r['transform']}", fontsize=11)
    V.stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(FIG / "fig4_best_cluster.png", dpi=120)
    plt.close(fig)


def fig5(s, tm, lab):
    rows = [0, 1, 5, 6, 8, 10, 13]
    fig, axs = plt.subplots(len(rows) + 2, 1, figsize=(12.5, 8.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.0, 0.8] + [0.8] * len(rows)))
    axs[0].plot(s.sec, s.load_kN, color=INK, lw=1.2)
    axs[0].set_ylabel("载荷 (kN)", fontsize=8)
    V.boundaries(axs[0], s)
    V.clean(axs[0])
    V.stage_bar(axs[1], s)
    for ax, i in zip(axs[2:], rows):
        r = tm[tm.id == i].iloc[0]
        lb = lab[r["transform"]].to_numpy()
        for v in sorted(set(lb)):
            ax.bar(s.sec[lb == v], 1, width=1.0, color=V.label_color(v), lw=0)
        ax.set_yticks([])
        ax.set_ylabel(short(r["transform"]), rotation=0, ha="right", va="center", fontsize=7.5)
        ax.text(1.005, 0.5, f"ARI {max(r.ARI_GMM, r.ARI_Ward):.2f}\n零模型 {r.null_pct:.0f}%",
                transform=ax.transAxes, fontsize=7, va="center")
        V.boundaries(ax, s)
        V.clean(ax, grid_y=False)
        ax.spines["left"].set_visible(False)
    axs[-1].set_xlabel("时间 (s)", fontsize=9)
    fig.suptitle("图 5  聚类标签沿时间展开（各变换取 GMM / Ward 中 ARI 较高者）", fontsize=11)
    V.stage_legend(fig, y=0.0)
    fig.tight_layout(rect=(0.04, 0.05, 0.93, 0.95))
    fig.savefig(FIG / "fig5_cluster_timeline.png", dpi=120)
    plt.close(fig)


def fig6(tm):
    null = pd.read_csv(HERE.parent / "E2_尺度无关DIC特征聚类" / "outputs" / "null_model.csv", index_col=0).iloc[:, 0]
    t = tm.copy()
    t["ARI"] = t[["ARI_GMM", "ARI_Ward"]].max(axis=1)
    t["name"] = [f"{i:02d} {short(n)}" for i, n in zip(t.id, t["transform"])]
    t = t.iloc[::-1]
    col = ["#8a8984" if n.startswith("REF") else "#1baf7a" if n.startswith("T4") else
           "#eb6834" if "Dong" in n else "#2a78d6" for n in t["transform"]]
    fig, axs = plt.subplots(1, 4, figsize=(17, 6.2), sharey=True)
    for ax, c, title, xl in ((axs[0], "silhouette_true", "人工阶段轮廓系数 ↑", (-0.1, 0.7)),
                             (axs[1], "fisher", "Fisher 比 = 阶段间 / 阶段内散度 ↑", None),
                             (axs[2], "intra_over_min_center", "阶段内平均半径 / 最近阶段中心距 ↓", None),
                             (axs[3], "ARI", "ARI（GMM / Ward 较高者）↑", (0, 1))):
        ax.barh(t.name, t[c], color=col)
        for i, v in enumerate(t[c]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)
        ax.set_title(title, fontsize=9)
        if xl:
            ax.set_xlim(*xl)
        V.clean(ax, grid_y=False)
        ax.tick_params(axis="y", labelsize=7.5)
    axs[3].axvline(float(null["95%"]), color=MUTED, ls="--", lw=1)
    axs[3].text(float(null["95%"]), len(t) - 0.4, " 零模型 95%", fontsize=7, color=MUTED)
    fig.legend([Patch(color=c) for c in ("#2a78d6", "#1baf7a", "#eb6834", "#8a8984")],
               ["单块变换（T0–T3）", "分块 MFA（T4）", "含 Dong 固定量程特征", "有监督参照"],
               frameon=False, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle("图 6  各变换的阶段聚合 / 离散指标与聚类 ARI", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIG / "fig6_transform_metrics.png", dpi=120)
    plt.close(fig)


def fig7(s):
    df = pd.read_csv(OUT / "dong_frame_levels.csv").dropna(subset=["p1_0"])
    fi = pd.read_csv(TE.E2C.F01.C.OUT / "frame_index.csv")[["frame", "t_s"]]
    df = df.merge(fi, on="frame")
    ev = json.loads((OUT / "dong_events.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.38, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    P = df[[f"p1_{i}" for i in range(32)]].to_numpy().T
    im = ax.imshow(np.log10(P + 1e-4), aspect="auto", origin="lower", cmap=V.SEQ, vmin=-4, vmax=0,
                   extent=[df.t_s.min(), df.t_s.max(), 0, 30])
    for x in (270, 334, 427.3):
        ax.axvline(x, color=INK, ls="--", lw=0.9)
    for lv in (6.56, 12.19):
        ax.axhline(lv, color="#eb6834", lw=0.8, ls=":")
        ax.text(df.t_s.min() + 2, lv + 0.3, f"{lv}‰", fontsize=7, color="#eb6834")
    ax.set_ylabel("ε1 等级（‰，固定量程 0–30‰ 分 32 级）", fontsize=8)
    ax.set_xlabel("时间 (s)", fontsize=8)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("log10(该级面积占比)", fontsize=8)
    ax.set_title("(a) Dong SCCM：ε1 各固定等级的面积占比随时间（虚线 = 人工阶段边界）", fontsize=9)
    V.clean(ax, grid_y=False)
    ax = fig.add_subplot(gs[1, 0])
    sc = pd.DataFrame(ev["criterion1_threshold_scan"])
    ax.plot(sc.theta_permil, sc.t_cross_2pct, color="#2a78d6", marker="o", lw=1.8)
    for x, nm in ((270, "人工起裂 270 s"), (334, "人工损伤 334 s")):
        ax.axhline(x, color=INK, ls="--", lw=0.8)
        ax.text(sc.theta_permil.min(), x + 3, nm, fontsize=7)
    ax.axvspan(5.39, 6.71, color="#eb6834", alpha=0.12, lw=0)
    ax.text(6.05, sc.t_cross_2pct.min() + 2, "Dong 原文\nεmd 5.4–6.7‰", fontsize=7, ha="center", color="#b8302f")
    ax.set_xlabel("阈值 θ（‰）", fontsize=8)
    ax.set_ylabel("P(ε1 ≥ θ) 首次 ≥ 2% 的时刻 (s)", fontsize=8)
    ax.set_title("(b) 判据 1 的阈值灵敏度：事件时刻随阈值几乎线性后移", fontsize=9)
    V.clean(ax)
    ax = fig.add_subplot(gs[1, 1])
    for c, col, nm in (("dong_P2_init", "#1baf7a", "ε2 初始等级占比（Dong 判据 2）"),
                       ("dong_P1_ge6", "#eb6834", "P(ε1 ≥ 6.56‰)"),
                       ("dong_level_entropy", "#8a8984", "ε1 32 级分布的归一化熵")):
        ax.plot(s.sec, s[c], color=col, lw=1.5, label=nm)
    for x in (270, 334, 427.3):
        ax.axvline(x, color=INK, ls="--", lw=0.8)
    ax.axvline(ev["criterion2_e2_init_below_2pct_s"], color="#1baf7a", ls=":", lw=1)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_title(f"(c) 逐秒 Dong 特征；ε2 初始等级首次 < 2% 于 {ev['criterion2_e2_init_below_2pct_s']:.0f} s", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    V.clean(ax)
    fig.suptitle("图 7  Dong SCCM 原文（固定量程）在 HS-2 上的复现", fontsize=11)
    fig.savefig(FIG / "fig7_dong_sccm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig8():
    de = pd.read_csv(OUT / "dong_feature_eval.csv")
    gc = {"Dong固定量程": "#eb6834", "增量集中": "#2a78d6", "增量运动学": "#86b6ef", "累积形态": "#8a8984"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axs[0]
    for g, col in gc.items():
        d = de[de.group == g]
        ax.scatter(d.rho_time.abs(), d.eps2, s=60, color=col, edgecolor="white", label=g, zorder=3)
        for r in d.itertuples():
            ax.annotate(r.feature, (abs(r.rho_time), r.eps2), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("与时间 |ρ|（越大越像时钟）", fontsize=8)
    ax.set_ylabel("与阶段的效应量 ε²（越大越能区分阶段）", fontsize=8)
    ax.set_title("(a) 区分阶段的能力 vs 携带时间结构的程度", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    V.clean(ax)
    ax.grid(color=GRID, lw=0.5)
    ax = axs[1]
    cols = [c for c in de.columns if c.startswith("sep_")]
    M = de.set_index("feature")[cols]
    ax.imshow(M.to_numpy(), cmap=V.SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [c[4:] for c in cols], fontsize=8)
    ax.set_yticks(range(M.shape[0]), [f"{f}  [{g}]" for f, g in zip(M.index, de.group)], fontsize=7.5)
    ax.set_title("(b) 相邻阶段区分度 |2·AUC − 1|", fontsize=9)
    V.clean(ax, grid_y=False)
    fig.suptitle("图 8  Dong 固定量程应变集中特征与 E2 增量场 / 累积形态特征的对比", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig8_dong_vs_inc.png", dpi=120)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    s = pd.read_csv(OUT / "samples.csv")
    tm = pd.read_csv(OUT / "transform_metrics.csv")
    lab = pd.read_csv(OUT / "cluster_labels.csv")
    fig3(s, tm)
    fig4(s, tm, lab)
    fig5(s, tm, lab)
    fig6(tm)
    fig7(s)
    fig8()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
