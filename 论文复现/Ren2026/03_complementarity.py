"""03 AE 与 DIC 特征在加载阶段上的表征能力与互补性（单试样 HS-2，回顾性）。

分析
  A 逐特征、逐阶段转换的区分度：|2·AUC−1|（起裂前↔起裂至损伤、起裂至损伤↔损伤至峰值、损伤至峰值↔峰后）
  B 分模态多分类：仅 AE / 仅 DIC / AE+DIC（直接拼接），外加"仅时间"基线
     - 特征集两套：全部；去掉 trend 类（累积量≈时钟）
     - 分类器两种：逻辑回归（线性）、随机森林（非线性）
     - 评估：阶段分层的分块交叉验证——每个阶段按时间切成 K 段，第 k 折测试集 = 各阶段第 k 段；
       训练集剔除与测试秒相距 < W 秒的样本（特征窗重叠）
  C 互补性：
     - 各阶段召回率对比（谁擅长哪个阶段）
     - 逐秒对错交叉表：AE 错而 DIC 对 / DIC 错而 AE 对；两者都错的比例 → 联合的理论上限
     - 联合 − 单模态 的平衡准确率差，分块 bootstrap 95% CI
     - 二分类（相邻阶段）上的模态对比
     - 冗余度：AE 与 DIC 特征的 Spearman 相关（原始、以及扣除阶段均值后的阶段内相关）
  D 无监督：KMeans(k=4) 的 ARI（与本项目自监督路线衔接）

输出 outputs/comp_*.csv, outputs/fig_feature_transition_auc.png, outputs/fig_stage_recall.png,
     outputs/fig_oof_timeline.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, binomtest

HERE = Path(__file__).resolve().parent
MS = HERE.parents[1] / "methodScripts"
sys.path.insert(0, str(MS))
import common as C                                    # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "outputs"
K_FOLDS = 5
N_BOOT = 1000
BOOT_BLOCK = 20
STAGES = [0, 1, 2, 3]
STAGE_ZH = ["起裂前", "起裂至损伤", "损伤至峰值", "峰后"]
PAIRS = [(0, 1), (1, 2), (2, 3)]
MOD_COLOR = {"AE": "#2a78d6", "DIC": "#eb6834", "AE+DIC": "#1baf7a", "时间": "#8a8984"}


# ------------------------------------------------------------------ 数据
def load(W):
    df = pd.read_csv(OUT / f"features_W{W}.csv")
    cat = pd.read_csv(OUT / "feature_catalog.csv")
    df = df[(df.stage_code >= 0) & (df.uncertain == 0)].reset_index(drop=True)
    return df, cat


def feature_sets(cat):
    ae = cat[cat.modality == "AE"]
    dic = cat[cat.modality == "DIC"]
    sets = {}
    for tag, keep in (("全部", lambda c: c), ("去趋势", lambda c: c[c.kind != "trend"])):
        sets[("AE", tag)] = list(keep(ae).feature)
        sets[("DIC", tag)] = list(keep(dic).feature)
        sets[("AE+DIC", tag)] = sets[("AE", tag)] + sets[("DIC", tag)]
    sets[("时间", "基线")] = ["sec"]
    return sets


def folds(df, W, k=K_FOLDS):
    """阶段分层分块：每个阶段按时间切 k 段；第 i 折测试 = 各阶段第 i 段；训练剔除距测试 < W 秒的样本。"""
    test_id = np.full(len(df), -1)
    for s in STAGES:
        idx = np.where(df.stage_code.to_numpy() == s)[0]
        for i, chunk in enumerate(np.array_split(idx, k)):
            test_id[chunk] = i
    sec = df.sec.to_numpy()
    out = []
    for i in range(k):
        te = np.where(test_id == i)[0]
        d = np.min(np.abs(sec[:, None] - sec[te][None, :]), axis=1)
        tr = np.where((test_id != i) & (d >= W))[0]
        out.append((tr, te))
    return out


def model(kind):
    if kind == "LR":
        clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                     class_weight="balanced", random_state=0, n_jobs=-1)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def oof_predict(df, cols, fl, kind, labels=None):
    y = df.stage_code.to_numpy() if labels is None else labels
    X = df[cols].to_numpy(float)
    pred = np.full(len(df), -1)
    proba = np.full(len(df), np.nan)
    for tr, te in fl:
        if len(np.unique(y[tr])) < 2:
            continue
        m = model(kind).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        if len(np.unique(y)) == 2:
            proba[te] = m.predict_proba(X[te])[:, 1]
    return pred, proba


def block_boot_diff(y, pa, pb, n=N_BOOT, block=BOOT_BLOCK, seed=0):
    """移动块 bootstrap：平衡准确率(pa) − 平衡准确率(pb) 的 95% CI。"""
    rng = np.random.default_rng(seed)
    N = len(y)
    starts = np.arange(0, N - block + 1)
    d = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, N // block + 1)])[:N]
        if len(np.unique(y[idx])) < 4:
            continue
        d.append(balanced_accuracy_score(y[idx], pa[idx]) - balanced_accuracy_score(y[idx], pb[idx]))
    return np.percentile(d, [2.5, 97.5])


# ------------------------------------------------------------------ A 逐特征区分度
def feature_transition_auc(df, cat):
    rows = []
    for r in cat.itertuples():
        for a, b in PAIRS:
            s = df[df.stage_code.isin([a, b])][[r.feature, "stage_code"]].dropna()
            if s.stage_code.nunique() < 2 or s[r.feature].nunique() < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(s.stage_code == b, s[r.feature])
            rows.append(dict(feature=r.feature, modality=r.modality, kind=r.kind,
                             transition=f"{STAGE_ZH[a]}→{STAGE_ZH[b]}", auc=auc, sep=abs(2 * auc - 1),
                             n_a=int((s.stage_code == a).sum()), n_b=int((s.stage_code == b).sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B/C 多分类与互补
def multiclass(df, sets, fl):
    y = df.stage_code.to_numpy()
    res, preds = [], {}
    for kind in ("LR", "RF"):
        for (mod, tag), cols in sets.items():
            p, _ = oof_predict(df, cols, fl, kind)
            ok = p >= 0
            preds[(kind, mod, tag)] = p
            rec = recall_score(y[ok], p[ok], labels=STAGES, average=None, zero_division=0)
            res.append(dict(clf=kind, modality=mod, feature_set=tag, n_features=len(cols),
                            bal_acc=balanced_accuracy_score(y[ok], p[ok]),
                            macro_f1=f1_score(y[ok], p[ok], average="macro"),
                            **{f"recall_{STAGE_ZH[s]}": rec[i] for i, s in enumerate(STAGES)}))
    return pd.DataFrame(res), preds


def complementarity(df, preds):
    y = df.stage_code.to_numpy()
    rows, ci_rows = [], []
    for kind in ("LR", "RF"):
        for tag in ("全部", "去趋势"):
            pa, pd_, pj = (preds[(kind, m, tag)] for m in ("AE", "DIC", "AE+DIC"))
            ca, cd, cj = pa == y, pd_ == y, pj == y
            n01, n10 = int((~ca & cd).sum()), int((ca & ~cd).sum())
            p_mcn = binomtest(n01, n01 + n10).pvalue if n01 + n10 else np.nan
            rows.append(dict(clf=kind, feature_set=tag,
                             both_correct=(ca & cd).mean(), AE_only_correct=(ca & ~cd).mean(),
                             DIC_only_correct=(~ca & cd).mean(), both_wrong=(~ca & ~cd).mean(),
                             oracle_acc=1 - (~ca & ~cd).mean(), joint_acc=cj.mean(),
                             AE_acc=ca.mean(), DIC_acc=cd.mean(), mcnemar_AE_vs_DIC_p=p_mcn))
            for nm, other in (("AE", pa), ("DIC", pd_)):
                lo, hi = block_boot_diff(y, pj, other)
                ci_rows.append(dict(clf=kind, feature_set=tag, comparison=f"AE+DIC − {nm}",
                                    diff=balanced_accuracy_score(y, pj) - balanced_accuracy_score(y, other),
                                    ci_low=lo, ci_high=hi))
            for s in STAGES:
                m = y == s
                rows[-1][f"AE错DIC对_{STAGE_ZH[s]}"] = float((~ca[m] & cd[m]).mean())
                rows[-1][f"DIC错AE对_{STAGE_ZH[s]}"] = float((ca[m] & ~cd[m]).mean())
    return pd.DataFrame(rows), pd.DataFrame(ci_rows)


def pairwise(df, sets, W):
    rows = []
    for a, b in PAIRS:
        sub = df[df.stage_code.isin([a, b])].reset_index(drop=True)
        fl = folds(sub.assign(stage_code=sub.stage_code), W)
        y = (sub.stage_code == b).astype(int).to_numpy()
        for (mod, tag), cols in sets.items():
            for kind in ("LR", "RF"):
                p, pr = oof_predict(sub, cols, fl, kind, labels=y)
                ok = p >= 0
                rows.append(dict(transition=f"{STAGE_ZH[a]}→{STAGE_ZH[b]}", modality=mod, feature_set=tag, clf=kind,
                                 auc=roc_auc_score(y[ok], pr[ok]), bal_acc=balanced_accuracy_score(y[ok], p[ok])))
    return pd.DataFrame(rows)


def redundancy(df, cat):
    ae = [c for c in cat[cat.modality == "AE"].feature]
    dic = [c for c in cat[cat.modality == "DIC"].feature]
    within = df.copy()
    for c in ae + dic:
        within[c] = df[c] - df.groupby("stage_code")[c].transform("mean")
    rows = []
    for a in ae:
        for d in dic:
            s = df[[a, d]].dropna()
            w = within[[a, d]].dropna()
            rows.append(dict(ae=a, dic=d, rho=spearmanr(s[a], s[d])[0], rho_within_stage=spearmanr(w[a], w[d])[0]))
    return pd.DataFrame(rows)


def unsupervised(df, sets):
    y = df.stage_code.to_numpy()
    rows = []
    for (mod, tag), cols in sets.items():
        X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(float))
        X = StandardScaler().fit_transform(X)
        Z = PCA(n_components=min(0.9, X.shape[1]) if X.shape[1] > 1 else 1, random_state=0).fit_transform(X) \
            if X.shape[1] > 1 else X
        ari = [adjusted_rand_score(y, KMeans(4, n_init=20, random_state=s).fit_predict(Z)) for s in range(5)]
        rows.append(dict(modality=mod, feature_set=tag, n_features=len(cols), pcs=Z.shape[1],
                         ARI_mean=np.mean(ari), ARI_std=np.std(ari)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 图
def _style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    return plt


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_auc_heatmap(fa, cat):
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f3f0", "#a9c8ef", "#2a78d6", "#123e7a"])
    order = list(cat.sort_values(["modality", "kind", "feature"]).feature)
    piv = fa.pivot(index="feature", columns="transition", values="sep").loc[order]
    piv = piv[[f"{STAGE_ZH[a]}→{STAGE_ZH[b]}" for a, b in PAIRS]]
    kinds = cat.set_index("feature").loc[order, "kind"]
    fig, ax = plt.subplots(figsize=(6.4, 10))
    ax.imshow(piv.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0]), [f"{f}  [{kinds[f]}]" for f in piv.index], fontsize=7)
    n_ae = int((cat.modality == "AE").sum())
    ax.axhline(n_ae - 0.5, color="#0b0b0b", lw=1.2)
    ax.text(piv.shape[1] - 0.45, n_ae / 2 - 0.5, "AE", rotation=90, va="center", fontsize=10, color=MOD_COLOR["AE"])
    ax.text(piv.shape[1] - 0.45, n_ae + (len(order) - n_ae) / 2 - 0.5, "DIC", rotation=90, va="center",
            fontsize=10, color=MOD_COLOR["DIC"])
    ax.set_xlim(-0.5, piv.shape[1] - 0.2)
    ax.set_title("单特征对相邻阶段的区分度 |2·AUC−1|（W=5 s）", fontsize=10)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_transition_auc.png", dpi=120)
    plt.close(fig)


def plot_recall(mc):
    plt = _style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, tag in zip(axs, ("全部", "去趋势")):
        sub = mc[(mc.clf == "RF") & ((mc.feature_set == tag) | (mc.modality == "时间"))]
        mods = ["时间", "AE", "DIC", "AE+DIC"]
        x = np.arange(len(STAGE_ZH) + 1)
        w = 0.2
        for i, m in enumerate(mods):
            r = sub[sub.modality == m].iloc[0]
            vals = [r[f"recall_{s}"] for s in STAGE_ZH] + [r.bal_acc]
            ax.bar(x + (i - 1.5) * w, vals, width=w - 0.02, color=MOD_COLOR[m], label=m)
        ax.set_xticks(x, STAGE_ZH + ["平衡准确率"], fontsize=8)
        ax.set_title(f"随机森林，特征集：{tag}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        _clean(ax)
    axs[0].set_ylabel("各阶段召回率", fontsize=9)
    axs[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage_recall.png", dpi=120)
    plt.close(fig)


def plot_timeline(df, preds, tag="去趋势"):
    plt = _style()
    rows = [("真实阶段", df.stage_code.to_numpy())] + [(m, preds[("RF", m, tag)]) for m in ("AE", "DIC", "AE+DIC")]
    stage_col = ["#c9dcf3", "#f6c9b3", "#b5e3cf", "#d9d8d3"]
    fig, axs = plt.subplots(len(rows) + 1, 1, figsize=(11, 4.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[2.2] + [1] * len(rows)))
    axs[0].plot(df.sec, df.load_kN, color="#0b0b0b", lw=1)
    axs[0].set_ylabel("载荷 kN", fontsize=8)
    _clean(axs[0])
    for ax, (nm, p) in zip(axs[1:], rows):
        for s in STAGES:
            m = p == s
            ax.bar(df.sec[m], 1, width=1.0, color=stage_col[s], lw=0)
        if nm != "真实阶段":
            wrong = p != df.stage_code.to_numpy()
            ax.bar(df.sec[wrong], 0.25, width=1.0, bottom=0.75, color="#0b0b0b", lw=0)
        ax.set_yticks([])
        ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
        _clean(ax)
    from matplotlib.patches import Patch
    axs[-1].set_xlabel("时间 (s)", fontsize=8)
    fig.legend([Patch(color=c) for c in stage_col] + [Patch(color="#0b0b0b")], STAGE_ZH + ["错判（黑顶）"],
               frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"逐秒交叉验证预测（随机森林，{tag}特征）", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_oof_timeline.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main
def run(W, plots):
    df, cat = load(W)
    sets = feature_sets(cat)
    fl = folds(df, W)
    fa = feature_transition_auc(df, cat)
    mc, preds = multiclass(df, sets, fl)
    comp, ci = complementarity(df, preds)
    pw = pairwise(df, sets, W)
    red = redundancy(df, cat)
    un = unsupervised(df, sets)
    for nm, t in (("feature_transition_auc", fa), ("multiclass", mc), ("error_overlap", comp),
                  ("joint_gain_ci", ci), ("pairwise", pw), ("redundancy", red), ("unsupervised", un)):
        t.to_csv(OUT / f"comp_{nm}_W{W}.csv", index=False, encoding="utf-8-sig")
    if plots:
        plot_auc_heatmap(fa, cat)
        plot_recall(mc)
        plot_timeline(df, preds)
    return df, cat, fa, mc, comp, ci, pw, red, un


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df, cat, fa, mc, comp, ci, pw, red, un = run(5, plots=True)
    print(f"样本：{len(df)} 秒（剔除不确定）；各阶段", df.stage_code.value_counts().sort_index().tolist())
    print("\n== A 每个阶段转换上，各模态最强单特征 |2AUC−1| ==")
    top = fa.sort_values("sep", ascending=False).groupby(["transition", "modality"]).head(1)
    print(top.sort_values(["transition", "modality"])[["transition", "modality", "feature", "kind", "auc", "sep"]]
          .round(3).to_string(index=False))
    print("\n  各转换上 sep ≥ 0.8 的特征数：")
    print(fa[fa.sep >= 0.8].groupby(["transition", "modality"]).size().unstack(fill_value=0).to_string())
    print("\n== B 四分类（交叉验证） ==")
    print(mc.round(3).to_string(index=False))
    print("\n== C1 逐秒对错交叉 ==")
    print(comp.round(3).T.to_string(header=False))
    print("\n== C2 联合增益（平衡准确率差，块 bootstrap 95% CI） ==")
    print(ci.round(3).to_string(index=False))
    print("\n== C3 相邻阶段二分类 AUC（RF / LR） ==")
    print(pw.pivot_table(index=["transition", "feature_set"], columns=["clf", "modality"], values="auc").round(3).to_string())
    print("\n== C4 AE–DIC 冗余度 ==")
    print("  |ρ| 中位数：原始 %.2f，阶段内 %.2f；|ρ|>0.8 的特征对：原始 %d，阶段内 %d / %d"
          % (red.rho.abs().median(), red.rho_within_stage.abs().median(), (red.rho.abs() > 0.8).sum(),
             (red.rho_within_stage.abs() > 0.8).sum(), len(red)))
    print("\n== D 无监督 KMeans(4) ARI ==")
    print(un.round(3).to_string(index=False))
    print("\n== 敏感性：W = 10 s ==")
    _, _, _, mc10, comp10, ci10, _, _, _ = run(10, plots=False)
    print(mc10[mc10.clf == "RF"][["modality", "feature_set", "bal_acc", "macro_f1"]].round(3).to_string(index=False))
    print(ci10.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
