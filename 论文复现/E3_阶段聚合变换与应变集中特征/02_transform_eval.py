"""02 E3 主实验：尺度无关的"阶段内聚合、阶段间离散"变换，以及 Dong 应变集中特征能否纳入表征。

所有变换都不使用阶段标签（无监督）；有监督 LDA 只作为"这些特征最多能分到什么程度"的参照上限。

一、变换（输入 = E2 的 DIC 增量场 6 个特征，除非注明）
  T0  E2 基线：全序列鲁棒 z → PCA 90%
  T1  噪声锚定 log-SNR：每个集中度特征除以"纯噪声理论值"再取对数（log(P2/0.0625) 等），
      inc_nu、inc_corr 原样；不使用任何数据统计量（跨试样时无需重新标定）
  T2  基线锚定：以加载初期 10–60 s（应变增量 ≈ DIC 噪声）为参照标准化 —— 因果
  T3  因果平滑：T0 后对每列做过去 W 秒滚动中位数（W = 10 / 30 s）
  T4  分块 MFA：特征按物理含义分块，每块鲁棒 z 后除以本块第一奇异值，再拼接 → PCA 90%
      块：A 增量集中 [inc_P2, inc_P4, inc_q98, inc_top5]；B 增量运动学 [inc_nu, inc_corr]；
          C 累积形态 [sf_tens, sf_nu, sf_e2low, sf_P4, sf_crack]；D Dong 固定量程 [dong_*]
      对比 A+B（= 增量场 6 个，按块等权）/ A+B+C / A+B+D / A+B+C+D
  T5  直接拼接对照：不分块，inc6 + Dong
  T6  慢特征分析 SFA（无监督，提取变化最慢的方向）：inc6 + 累积形态
  REF 有监督 LDA（分块交叉验证），只作上限参照

二、指标（每个变换）
  聚合/离散：人工阶段在变换空间的轮廓系数（主）、Fisher 比 = 阶段间散度 / 阶段内散度、
            阶段内平均距离 / 阶段中心间最小距离
  聚类：GMM（5 种子平均）、Ward 的 ARI；零模型百分位；与时间 |ρ|

三、Dong 应变集中特征评估
  单特征：与阶段的效应量 ε²、与时间 ρ、相邻阶段 |2AUC−1|
  组合：上面 T4/T5 中加入 D 块后的变化

输出 outputs/transform_metrics.csv, outputs/embed_<变换>.csv, outputs/dong_feature_eval.csv,
     outputs/cluster_labels.csv, outputs/contingency_best.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
E1 = HERE.parent / "E1_单模态聚类基线"
E2 = HERE.parent / "E2_尺度无关DIC特征聚类"
OUT = HERE / "outputs"


def load_by_path(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CB = load_by_path(E1 / "02_cluster_baseline.py", "e1_cluster")
E2C = load_by_path(E2 / "02_cluster_baseline.py", "e2_cluster")
STAGE_ZH = ["起裂前", "起裂至损伤", "损伤至峰值", "峰后"]
A = ["inc_P2", "inc_P4", "inc_q98", "inc_top5"]
B = ["inc_nu", "inc_corr"]
Cc = ["sf_tens", "sf_nu", "sf_e2low", "sf_P4", "sf_crack"]
D = ["dong_P1_ge3", "dong_P1_ge6", "dong_P1_ge12", "dong_P2_init", "dong_level_entropy"]
NOISE = {"inc_P2": 0.0625, "inc_P4": 1.5e-5, "inc_q98": 1.25, "inc_top5": 0.112}
BASE = (10, 60)


# ------------------------------------------------------------------ 变换
def rz(X: pd.DataFrame):
    X = X.fillna(X.median())
    q = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
    q = q.where(q > 0, X.std()).where(lambda v: v > 0, 1.0)
    return ((X - X.median()) / q).clip(-5, 5)


def pca90(Z):
    Z = np.asarray(Z, float)
    if Z.shape[1] == 1:
        return Z
    return PCA(0.9, svd_solver="full", random_state=0).fit_transform(Z)


def log_snr(X):
    return pd.DataFrame({
        "inc_P2": np.log(np.maximum(X.inc_P2, 1e-4) / NOISE["inc_P2"]),
        "inc_P4": np.log((X.inc_P4 + 1e-3) / 1e-3),
        "inc_q98": X.inc_q98 - NOISE["inc_q98"],
        "inc_top5": np.log(X.inc_top5 / NOISE["inc_top5"]),
        "inc_nu": X.inc_nu, "inc_corr": X.inc_corr})


def mfa(s, blocks):
    Zs = []
    for b in blocks:
        Z = rz(s[b]).to_numpy()
        Z = Z - Z.mean(0)
        Zs.append(Z / np.linalg.svd(Z, compute_uv=False)[0])
    return pca90(np.hstack(Zs))


def sfa(Z, n=3):
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    keep = S > 1e-6 * S[0]
    Wh = Z @ Vt[keep].T / S[keep] * np.sqrt(len(Z))
    _, vec = np.linalg.eigh(np.cov(np.diff(Wh, axis=0).T))
    return Wh @ vec[:, :n]


def lda_cv(Z, y, mask, sec, k=5, gap=5):
    """阶段分层的分块交叉验证：每个阶段按时间切 k 段，第 i 折测试 = 各阶段第 i 段（含不确定秒按就近阶段归段），
    训练剔除距测试 < gap 秒的样本。保证每折训练集都包含 4 个阶段。"""
    test_id = np.full(len(Z), -1)
    lab = np.where(y >= 0, y, 0)
    for st in range(4):
        idx = np.where(lab == st)[0]
        for i, chunk in enumerate(np.array_split(idx, k)):
            test_id[chunk] = i
    out = np.zeros((len(Z), 3))
    for i in range(k):
        te = np.where(test_id == i)[0]
        d = np.min(np.abs(sec[:, None] - sec[te][None, :]), axis=1)
        tr = np.where(mask & (test_id != i) & (d >= gap))[0]
        out[te] = LinearDiscriminantAnalysis(n_components=3).fit(Z[tr], y[tr]).transform(Z[te])
    return out


def transforms(s, y, mask, sec):
    X6 = s[A + B]
    T = {}
    T["T0 E2 基线（增量6，全序列鲁棒z）"] = pca90(rz(X6))
    T["T1 噪声锚定 log-SNR"] = pca90(log_snr(X6).fillna(log_snr(X6).median()))
    L = log_snr(X6)
    b = (sec >= BASE[0]) & (sec <= BASE[1])
    T["T2 基线锚定（10–60 s 噪声期，因果）"] = pca90(((L - L[b].median()) / L[b].std()).fillna(0).clip(-50, 50))
    for Wn in (10, 30):
        T[f"T3 因果滚动中位数 {Wn} s"] = pca90(rz(X6.rolling(Wn, min_periods=1).median()))
    T["T4 MFA [A 增量集中 | B 增量运动学]"] = mfa(s, [A, B])
    T["T4 MFA [A | inc_nu | inc_corr]（3 块）"] = mfa(s, [A, ["inc_nu"], ["inc_corr"]])
    T["T4 MFA [A | B | C 累积形态]"] = mfa(s, [A, B, Cc])
    T["T4 MFA [A | B | D Dong固定量程]"] = mfa(s, [A, B, D])
    T["T4 MFA [A | B | C | D]"] = mfa(s, [A, B, Cc, D])
    T["T5 直接拼接 增量6 + Dong（不分块）"] = pca90(rz(s[A + B + D]))
    T["T5 仅 Dong 固定量程 5 个"] = pca90(rz(s[D]))
    T["T6 SFA 3 个最慢方向（增量6 + 累积形态）"] = sfa(rz(s[A + B + Cc]))
    T["REF 有监督 LDA（分块交叉验证，上限参照）"] = lda_cv(rz(s[A + B + Cc + D]).to_numpy(), y, mask, sec)
    return T


# ------------------------------------------------------------------ 指标
def scatter_metrics(E, y, mask):
    Em, ym = E[mask], y[mask]
    mu = Em.mean(0)
    cents = np.array([Em[ym == k].mean(0) for k in range(4)])
    within = sum(((Em[ym == k] - cents[k]) ** 2).sum() for k in range(4))
    between = sum((ym == k).sum() * ((cents[k] - mu) ** 2).sum() for k in range(4))
    intra = np.mean([np.linalg.norm(Em[ym == k] - cents[k], axis=1).mean() for k in range(4)])
    dc = [np.linalg.norm(cents[i] - cents[j]) for i in range(4) for j in range(i + 1, 4)]
    adj = [np.linalg.norm(cents[i] - cents[i + 1]) for i in range(3)]
    return dict(silhouette_true=silhouette_score(Em, ym), fisher=between / within,
                intra_over_min_center=intra / min(dc), intra_over_min_adjacent=intra / min(adj))


def evaluate(name, E, y, mask, sec, null):
    E = np.asarray(E, float)
    gm = [GaussianMixture(4, covariance_type="full", n_init=5, random_state=sd, reg_covar=1e-4).fit_predict(E)
          for sd in range(5)]
    aris = [adjusted_rand_score(y[mask], c[mask]) for c in gm]
    wd = AgglomerativeClustering(4, linkage="ward").fit_predict(E)
    a_w = adjusted_rand_score(y[mask], wd[mask])
    best = gm[int(np.argmax(aris))] if np.mean(aris) >= a_w else wd
    rho = np.median([abs(spearmanr(E[:, i], sec)[0]) for i in range(E.shape[1])])
    sw = sec[np.where(np.diff(best) != 0)[0] + 1]
    off = E2C.boundary_offset(sec, best)
    r = dict(transform=name, dims=E.shape[1], **scatter_metrics(E, y, mask),
             ARI_GMM=float(np.mean(aris)), ARI_GMM_std=float(np.std(aris)), ARI_Ward=a_w,
             null_pct=float((null < max(np.mean(aris), a_w)).mean() * 100),
             sil_cluster=silhouette_score(E[mask], best[mask]), n_switches=len(sw),
             off_ci_s=off[0], off_cd_s=off[1], off_peak_s=off[2], rho_time=rho)
    return r, best


def dong_eval(s, y, mask, sec):
    rows = []
    for f in D + A + B + Cc:
        x = s[f]
        g = [x[mask & (y == k)].dropna() for k in range(4)]
        H = kruskal(*g).statistic
        r = dict(feature=f, group=("Dong固定量程" if f in D else "增量集中" if f in A else "增量运动学" if f in B
                                   else "累积形态"),
                 eps2=(H - 3) / (mask.sum() - 4), rho_time=spearmanr(x.dropna(), sec[x.notna()])[0])
        for a, b in ((0, 1), (1, 2), (2, 3)):
            sub = mask & np.isin(y, [a, b]) & x.notna().to_numpy()
            r[f"sep_{STAGE_ZH[a]}→{STAGE_ZH[b]}"] = abs(2 * roc_auc_score(y[sub] == b, x[sub]) - 1)
        for k in range(4):
            r[f"med_{STAGE_ZH[k]}"] = float(g[k].median())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    s = pd.read_csv(OUT / "samples.csv")
    y = s.stage_code.to_numpy()
    mask = (y >= 0) & (s.uncertain.to_numpy() == 0)
    sec = s.sec.to_numpy()
    null = E2C.contiguous_null(len(s), np.where(mask)[0], y)

    T = transforms(s, y, mask, sec)
    rows, labels = [], {"sec": sec, "stage_code": y, "uncertain": s.uncertain}
    for i, (name, E) in enumerate(T.items()):
        r, best = evaluate(name, E, y, mask, sec, null)
        rows.append(r)
        labels[name] = CB.align_to_stages(np.where(mask, y, -1), best)
        emb = pd.DataFrame(np.asarray(E)[:, :min(3, np.asarray(E).shape[1])],
                           columns=[f"C{j + 1}" for j in range(min(3, np.asarray(E).shape[1]))])
        emb.insert(0, "sec", sec)
        emb.to_csv(OUT / f"embed_{i:02d}.csv", index=False, encoding="utf-8-sig")
    tm = pd.DataFrame(rows)
    tm.insert(0, "id", range(len(tm)))
    tm.to_csv(OUT / "transform_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(labels).to_csv(OUT / "cluster_labels.csv", index=False, encoding="utf-8-sig")

    de = dong_eval(s, y, mask, sec)
    de.to_csv(OUT / "dong_feature_eval.csv", index=False, encoding="utf-8-sig")

    unsup = tm[~tm["transform"].str.startswith("REF")]
    best = unsup.loc[unsup[["ARI_GMM", "ARI_Ward"]].max(axis=1).idxmax()]
    ct = pd.crosstab(pd.Series(y[mask], name="人工阶段").map(dict(enumerate(STAGE_ZH))),
                     pd.Series(np.asarray(labels[best["transform"]])[mask], name="簇（已对齐）")).reindex(STAGE_ZH)
    ct.to_csv(OUT / "contingency_best.csv", encoding="utf-8-sig")

    print("== 变换对比（阶段聚合/离散：轮廓系数、Fisher 比越大越好，intra/min 越小越好） ==")
    print(tm.drop(columns=["ARI_GMM_std"]).round(3).to_string(index=False))
    print(f"\n最佳无监督变换：{best['transform']}；列联表：\n{ct.to_string()}")
    print("\n== Dong 固定量程特征 vs 其他组（ε² / 与时间 ρ / 相邻阶段 |2AUC−1|） ==")
    print(de.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
