"""03 每帧 vs 滑动窗口：DIC 特征的时间尺度分析与窗口高维特征构建。

问题：是否逐帧单独处理即可，还是需要把多帧组成滑动窗口提取高维特征？
从四个角度量化回答：
  A 噪声与速率：每个特征的单帧噪声 σ；Ren 方法核心的相对张拉/滑移速率，在逐帧差分与 W 秒窗斜率下的信噪比
  B Ren 事件判别：局部化起始 t_ini、加速演化区间对 W 的稳定性
  C 阶段可分性：逐帧状态特征 vs 窗口（状态 + 动态）特征，对人工阶段的 KMeans-ARI / 轮廓系数，
    并给出“只用时间”的基线（阶段在时间上连续，单调特征天然跟时间走）
  D 去趋势检验：只用窗口动态特征（斜率、窗内波动、曲率），排除“特征≈时钟”的平凡解

窗口：步长 1 s，与人工标注 1 s 分辨率一致；窗口尾端对齐每个 1 s 标注格的末端（因果，可在线），长 W 秒。

输出 outputs/window_features_W{W}.csv, outputs/rate_snr.csv, outputs/ren_events.csv,
     outputs/separability.csv, outputs/ren_tini_sensitivity.csv, outputs/fig_overview.png
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler

import common as C

warnings.filterwarnings("ignore", category=RuntimeWarning)
WINDOWS_S = (1, 2, 5, 10)
SECS = (1, 2, 3)
# 状态特征（逐帧可得）。C_band、N_cc 在当前阈值下恒为 1，不入选
STATE = ["eyy_med", "exx_med", "ev2d_med", "gmax_p50", "gmax_lift", "gmax_cv", "F_e1_005", "A_loc",
         "LW", "width_mm", "rho_in_out", "angle_deg", "crack_ratio", "U_surr", "tau_band",
         *[f"{k}_net_{j}" for j in SECS for k in ("dn", "ds")]]
REN_DISP = [f"{k}_net_{j}" for j in SECS for k in ("dn", "ds")]
BASE_T = (5.0, 60.0)          # 基线时段（载荷 <~10% 峰值，起裂前早期）


# ------------------------------------------------------------------ 公用
def load_frames():
    f = pd.read_csv(C.OUT / "frame_features.csv")
    f = f[(f.stage_code >= 0)].reset_index(drop=True)
    return f


def frame_noise(f, cols):
    """单帧噪声：局部二阶差分的鲁棒标准差（对平滑趋势不敏感），x[k]-2x[k+1]+x[k+2] 方差 = 6σ²。"""
    pre = f[f.stage_code < 3]
    return {c: float((pre[c].diff().diff()).abs().median() / 0.6745 / np.sqrt(6)) for c in cols}


def window_slope(t, x, W):
    """尾窗 [t-W, t] 内线性回归斜率（逐帧）。W=0 → 相邻帧差分。"""
    if W == 0:
        return np.r_[np.nan, np.diff(x) / np.diff(t)]
    n = int(round(W * C.FPS)) + 1
    s = pd.Series(x)
    tt = pd.Series(t)
    mt, mx = tt.rolling(n).mean(), s.rolling(n).mean()
    cov = (tt * s).rolling(n).mean() - mt * mx
    var = (tt * tt).rolling(n).mean() - mt * mt
    return (cov / var).to_numpy()


def stage_bins(b):
    """1 s 标注格：[s, s+1)，标签取格中心；与 jiazaiSplit 窗口一致。"""
    tm, _, _ = C.read_load()
    s = np.arange(0, int(np.floor(tm[-1])))
    code, unc = C.label_times(s + 0.5, b)
    return pd.DataFrame({"sec": s, "t_end": s + 1.0, "stage_code": code, "uncertain": unc.astype(int)})


# ------------------------------------------------------------------ A 速率信噪比
def rate_snr(f, b):
    t = f.t_s.to_numpy()
    rows = []
    for c in REN_DISP:
        x = f[c].to_numpy()
        for W in (0, *WINDOWS_S):
            r = window_slope(t, x, W)
            base = (t >= BASE_T[0]) & (t <= BASE_T[1])
            sig_noise = np.nanstd(r[base])                       # 基线时段速率的离散 = 估计噪声
            for st, name in ((1, "S1"), (2, "S2")):
                m = f.stage_code.to_numpy() == st
                rows.append(dict(feature=c, W_s=W, stage=name, rate_median_mm_s=np.nanmedian(np.abs(r[m])),
                                 noise_std_mm_s=sig_noise, snr=np.nanmedian(np.abs(r[m])) / sig_noise))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B Ren 事件
def runs(mask, t, min_dur):
    """mask 为 True 且持续 ≥ min_dur 秒的区间列表。"""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end = i if v else i - 1
            if t[end] - t[start] >= min_dur - 1e-9:
                out.append((float(t[start]), float(t[end])))
            start = None
    return out


def ren_events(f, b, W, k_sigma=5.0, min_dur=5.0, k_mad=5.0):
    """Ren 判据的可复现版本（速率判据）。
    原文：带内外“同时开始出现相对位移增长”= 局部化起始；速率陡增 = 加速演化。
    起始 t_ini：≥2 个断面的相对张拉或滑移速率（W 秒窗斜率）偏离基线时段 > k_sigma·σ，持续 ≥ min_dur 秒的最早时刻。
      （注：直接对位移设 3σ 阈值会被 μm 级基线漂移在 70–100 s 触发，不可用）
    加速：平均相对位移合量 |δ| 的速率 > r0 + k_mad·MAD（r0、MAD 取 t_ini 至峰值），持续 ≥1 s。
    W=0 为逐帧差分。
    """
    t = f.t_s.to_numpy()
    base = (t >= BASE_T[0]) & (t <= BASE_T[1])
    cnt = np.zeros(len(t))
    for j in SECS:
        e = np.zeros(len(t), bool)
        for k in ("dn", "ds"):
            r = window_slope(t, f[f"{k}_net_{j}"].to_numpy(), W)
            mu, sd = np.nanmean(r[base]), np.nanstd(r[base])
            e |= np.abs(r - mu) > k_sigma * sd
        cnt += e
    rr = runs(cnt >= 2, t, min_dur)
    t_ini = rr[0][0] if rr else np.nan
    mag = np.mean([np.hypot(f[f"dn_net_{j}"], f[f"ds_net_{j}"]) for j in SECS], axis=0)
    rate = window_slope(t, mag, W)
    pre = (t >= t_ini) & (t < b["t_peak"])
    if pre.any():
        r0 = np.nanmedian(rate[pre])
        mad = 1.4826 * np.nanmedian(np.abs(rate[pre] - r0))
        acc = [a for a in runs(rate > r0 + k_mad * mad, t, 1.0) if a[0] > t_ini]
    else:
        r0 = mad = np.nan
        acc = []
    return dict(W_s=W, t_ini=t_ini, r0_um_s=r0 * 1e3, mad_um_s=mad * 1e3, n_acc=len(acc),
                acc_intervals=json.dumps([[round(x, 1) for x in a] for a in acc[:8]]))


def ren_sensitivity(f, b):
    rows = []
    for W in (1, 2, 5, 10):
        for ks in (3.0, 5.0, 8.0):
            for md in (2.0, 5.0):
                e = ren_events(f, b, W, k_sigma=ks, min_dur=md)
                rows.append(dict(W_s=W, k_sigma=ks, min_dur_s=md, t_ini=e["t_ini"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C/D 窗口特征
def window_features(f, bins, W):
    """每个 1 s 标注格 → 尾窗 W 秒内的：均值、末值（状态）；斜率、窗内残差标准差、二次项（动态）。"""
    t = f.t_s.to_numpy()
    X = f[STATE].to_numpy(float)
    rows = []
    for _, bn in bins.iterrows():
        m = (t > bn.t_end - W) & (t <= bn.t_end)
        if m.sum() < max(3, int(0.6 * W * C.FPS)):
            continue
        tt = t[m] - t[m].mean()
        r = dict(sec=bn.sec, stage_code=bn.stage_code, uncertain=bn.uncertain, n_frames=int(m.sum()))
        for i, c in enumerate(STATE):
            x = X[m, i]
            ok = np.isfinite(x)
            if ok.sum() < 3:
                continue
            p = np.polyfit(tt[ok], x[ok], 2 if ok.sum() >= 5 else 1)
            lin = np.polyfit(tt[ok], x[ok], 1)
            r[f"{c}__mean"] = x[ok].mean()
            r[f"{c}__slope"] = lin[0]
            r[f"{c}__resid_std"] = np.std(x[ok] - np.polyval(lin, tt[ok]))
            r[f"{c}__curv"] = p[0] if len(p) == 3 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def separability(X, y, seed=0):
    X = RobustScaler().fit_transform(X)
    X = np.clip(X, -10, 10)
    Z = PCA(n_components=0.95, random_state=seed).fit_transform(X)
    km = KMeans(4, n_init=20, random_state=seed).fit(Z)
    n = len(y)
    idx = np.random.default_rng(seed).choice(n, min(n, 2000), replace=False)
    return adjusted_rand_score(y, km.labels_), silhouette_score(Z[idx], y[idx]), Z.shape[1]


MORPH = ["LW", "width_mm", "rho_in_out", "angle_deg"]     # 成带形态：未成带时逐帧极不稳定


def run_separability(f, bins):
    res = []
    keep = (f.uncertain == 0) & (f.stage_code >= 0)
    ff = f[keep]
    y = ff.stage_code.to_numpy()
    for name, cols in (("逐帧 状态特征(全部)", STATE), ("逐帧 状态特征(不含形态)", [c for c in STATE if c not in MORPH])):
        Xf = ff[cols].to_numpy(float)
        ok = np.isfinite(Xf).all(1)
        a, s, d = separability(Xf[ok], y[ok])
        res.append(dict(rep=name, W_s=0.2, n=int(ok.sum()), dims_in=Xf.shape[1], pcs=d, ARI=a, silhouette=s))
    a, s, d = separability(ff[["t_s"]].to_numpy()[ok], y[ok])
    res.append(dict(rep="基线：仅时间（逐帧）", W_s=0.2, n=int(ok.sum()), dims_in=1, pcs=d, ARI=a, silhouette=s))
    for W in WINDOWS_S:
        wf = window_features(f, bins, W)
        wf.to_csv(C.OUT / f"window_features_W{W}.csv", index=False, encoding="utf-8-sig")
        wf = wf[(wf.uncertain == 0) & (wf.stage_code >= 0)]
        y = wf.stage_code.to_numpy()
        groups = {
            "窗口 状态(均值)": [c for c in wf.columns if c.endswith("__mean")],
            "窗口 状态(均值,不含形态)": [c for c in wf.columns if c.endswith("__mean") and c.split("__")[0] not in MORPH],
            "窗口 状态+动态": [c for c in wf.columns if "__" in c],
            "窗口 仅动态(去趋势)": [c for c in wf.columns if c.endswith(("__slope", "__resid_std", "__curv"))],
        }
        for name, cols in groups.items():
            X = wf[cols].to_numpy(float)
            ok = np.isfinite(X).all(1)
            a, s, d = separability(X[ok], y[ok])
            res.append(dict(rep=name, W_s=W, n=int(ok.sum()), dims_in=len(cols), pcs=d, ARI=a, silhouette=s))
        a, s, d = separability(wf[["sec"]].to_numpy(float), y)
        res.append(dict(rep="基线：仅时间（窗口）", W_s=W, n=len(y), dims_in=1, pcs=d, ARI=a, silhouette=s))
    return pd.DataFrame(res)


def dynamic_feature_ranking(W=5):
    """去趋势动态特征的逐特征阶段区分度（Kruskal–Wallis H 的效应量 ε²），帮助挑选。"""
    from scipy.stats import kruskal
    wf = pd.read_csv(C.OUT / f"window_features_W{W}.csv")
    wf = wf[(wf.uncertain == 0) & (wf.stage_code >= 0)]
    rows = []
    for c in [c for c in wf.columns if "__" in c]:
        g = [wf.loc[wf.stage_code == s, c].dropna().to_numpy() for s in range(4)]
        if min(len(x) for x in g) < 5:
            continue
        H = kruskal(*g).statistic
        n = sum(len(x) for x in g)
        rows.append(dict(feature=c, kind=c.split("__")[1], eps2=(H - 3) / (n - 4)))
    return pd.DataFrame(rows).sort_values("eps2", ascending=False)


# ------------------------------------------------------------------ 图
def plots(f, b, ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    S = ["#2a78d6", "#eb6834", "#1baf7a"]
    t = f.t_s.to_numpy()
    bounds = [b["t_ci"], b["t_cd"], b["t_peak"]]

    def deco(ax, ylabel):
        for x in bounds:
            ax.axvline(x, color=muted, lw=0.8, ls="--")
        ax.set_ylabel(ylabel, color=ink, fontsize=9)
        ax.grid(axis="y", color=grid, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(t, f.load_kN, color=ink, lw=1.2)
    for x0, x1, nm in ((0, b["t_ci"], "起裂前"), (b["t_ci"], b["t_cd"], "起裂至损伤"),
                       (b["t_cd"], b["t_peak"], "损伤至峰值"), (b["t_peak"], t.max(), "峰后")):
        axs[0].text((x0 + x1) / 2, f.load_kN.max() * 0.08, nm, ha="center", fontsize=9, color=muted)
    deco(axs[0], "载荷 (kN)")
    axs[1].plot(t, f.gmax_lift * 1e3, color=S[0], lw=1.2, label="γmax P95−P50")
    axs[1].plot(t, f.gmax_p50 * 1e3, color=S[1], lw=1.2, label="γmax P50")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    deco(axs[1], "γmax (‰)")
    axs[2].plot(t, f.F_e1_005 * 100, color=S[0], lw=1.2, label="e1≥0.005 面积")
    axs[2].plot(t, f.A_loc * 100, color=S[1], lw=1.2, label="局部化掩膜面积")
    axs[2].plot(t, f.crack_ratio * 100, color=S[2], lw=1.2, label="失配(裂缝)面积")
    axs[2].legend(frameon=False, fontsize=8, loc="upper left")
    deco(axs[2], "面积占比 (%)")
    for j in SECS:
        axs[3].plot(t, f[f"dn_net_{j}"], color=S[j - 1], lw=1.2, label=f"断面{j}")
        axs[4].plot(t, f[f"ds_net_{j}"], color=S[j - 1], lw=1.2, label=f"断面{j}")
    axs[3].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    deco(axs[3], "相对张拉 δn (mm)")
    deco(axs[4], "相对滑移 δs (mm)")
    mag = np.mean([np.hypot(f[f"dn_net_{j}"], f[f"ds_net_{j}"]) for j in SECS], axis=0)
    axs[5].plot(t, window_slope(t, mag, 0) * 1e3, color="#c3c2b7", lw=0.6, label="逐帧差分")
    axs[5].plot(t, window_slope(t, mag, 2) * 1e3, color=S[0], lw=1.2, label="2 s 窗斜率")
    axs[5].plot(t, window_slope(t, mag, 5) * 1e3, color=S[1], lw=1.2, label="5 s 窗斜率")
    e5 = ev[ev.W_s == 5].iloc[0]
    if np.isfinite(e5.t_ini):
        axs[5].axvline(e5.t_ini, color=S[2], lw=1.2)
    axs[5].set_ylim(-20, 60)
    if np.isfinite(e5.t_ini):
        axs[5].text(e5.t_ini - 2, 50, "局部化起始 t_ini (5 s 窗)", color=ink, fontsize=8, ha="right")
    for a0, a1 in json.loads(e5.acc_intervals):
        axs[5].axvspan(a0, a1, color="#eb6834", alpha=0.12, lw=0)
    axs[5].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    deco(axs[5], "|δ| 速率 (μm/s)")
    axs[5].set_xlabel("力学时钟时间 t (s)", fontsize=9)
    fig.suptitle("HS-2 DIC 逐帧特征与人工阶段（虚线：起裂 / 损伤 / 峰值）", fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(C.OUT / "fig_overview.png", dpi=110)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    f = load_frames()
    b = C.stage_boundaries()
    bins = stage_bins(b)

    noise = frame_noise(f, STATE)
    pd.Series(noise, name="frame_noise_sigma").to_csv(C.OUT / "frame_noise.csv", encoding="utf-8-sig")

    rs = rate_snr(f, b)
    rs.to_csv(C.OUT / "rate_snr.csv", index=False, encoding="utf-8-sig")
    print("== A 速率信噪比（Ren 相对位移速率，6 条曲线中位数） ==")
    print(rs.groupby(["W_s", "stage"]).agg(rate_um_s=("rate_median_mm_s", lambda x: 1e3 * x.median()),
                                            noise_um_s=("noise_std_mm_s", lambda x: 1e3 * x.median()),
                                            snr=("snr", "median")).unstack("stage").round(2).to_string())

    ev = pd.DataFrame([ren_events(f, b, W) for W in (0, *WINDOWS_S)])
    ev.to_csv(C.OUT / "ren_events.csv", index=False, encoding="utf-8-sig")
    sens = ren_sensitivity(f, b)
    sens.to_csv(C.OUT / "ren_tini_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n== B Ren 事件对窗口长度的稳定性 ==  (起裂 %.0f s, 损伤 %.0f s, 峰值 %.1f s)" % (b["t_ci"], b["t_cd"], b["t_peak"]))
    print(ev.to_string(index=False))
    print("  t_ini 灵敏度（k_sigma × 持续时间）：")
    print(sens.pivot_table(index="W_s", columns=["k_sigma", "min_dur_s"], values="t_ini").round(1).to_string())

    sep = run_separability(f, bins)
    sep.to_csv(C.OUT / "separability.csv", index=False, encoding="utf-8-sig")
    print("\n== C/D 阶段可分性（排除边界 ±1 s 与峰值不确定区间） ==")
    print(sep.round(3).to_string(index=False))

    rk = dynamic_feature_ranking(5)
    rk.to_csv(C.OUT / "dynamic_feature_ranking_W5.csv", index=False, encoding="utf-8-sig")
    print("\n== 5 s 窗口：各类窗口统计量的阶段区分度 ε²（中位数 / 前 3） ==")
    for kind, g in rk.groupby("kind"):
        print(f"  {kind:10s} median ε²={g.eps2.median():.3f}  top: " +
              ", ".join(f"{r.feature}({r.eps2:.2f})" for r in g.head(3).itertuples()))
    plots(f, b, ev)


if __name__ == "__main__":
    main()
