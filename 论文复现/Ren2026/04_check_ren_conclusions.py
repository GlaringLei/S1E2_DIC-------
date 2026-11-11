"""04 逐条检验 Ren et al. (2026) 关于 DIC 局部化与 AE 的结论在 HS-2 上是否成立。

  H1 局部化起始后 AE 事件率骤降（原文：636–659 s 骤降并维持低值，640 s 局部化起始）
  H2 局部化加速演化时 AE 事件率激增，剪切型占比显著升高（原文：18.33% → 35.17% → 40.54%）
  H3 AE 平静期内局部化带相对位移不停止，且平静期末段开始加速，各断面不同步（原文：779–797 s）
  H4 围岩应变能峰值接近 AE 率峰值，位于局部化起始后、加速前；平静期内剪应力几乎不变，
     平静期结束后剪应力陡降；能量释放先于应力释放（原文：U 峰 672–701 s）
  另：全程 DIC 速率与 AE 率的滞后互相关（检验“对应良好”的整体表述）

DIC 事件取自 methodScripts/outputs/ren_events.csv（5 s 窗，已做窗长/阈值灵敏度）。
AE 事件率 = 撞击数（4 通道合计，非定位事件）；HS-2 的 AE 只有整秒分辨率。
输出 outputs/hypothesis_checks.csv, outputs/ren_stage_table.csv, outputs/h1–h4_*.csv,
     outputs/xcorr.csv, outputs/xcorr_detrended.csv, outputs/fig_ren_checks.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
MS = HERE.parents[1] / "methodScripts"
sys.path.insert(0, str(MS))
import common as C                                    # noqa: E402

OUT = HERE / "outputs"
NORM = "minmax"            # 主口径：与 Ren“先归一化”一致；p99/median 作灵敏度
NORMS = ("minmax", "p99", "median")
SECS = (1, 2, 3)


def load():
    ae = pd.read_csv(OUT / "ae_seconds.csv")
    dic = pd.read_csv(OUT / "dic_seconds.csv")
    m = ae.merge(dic, on="sec", how="left")
    ev = pd.read_csv(MS / "outputs" / "ren_events.csv")
    e5 = ev[ev.W_s == 5].iloc[0]
    acc = [a for a in json.loads(e5.acc_intervals)]
    quiet = json.loads((OUT / "ae_quiet_periods.json").read_text(encoding="utf-8"))
    return m, float(e5.t_ini), acc, quiet


def mean_in(m, col, a, b):
    s = m.loc[(m.sec >= a) & (m.sec < b), col]
    return float(s.mean()) if len(s) else np.nan


def shear_frac(m, a, b, k=NORM):
    s = m[(m.sec >= a) & (m.sec < b)]
    return float(s[f"shear_hits_{k}"].sum() / max(s.hits.sum(), 1))


def ren_stages(t_ini, acc, b):
    """按 Ren 图 11 的六段划分构造 HS-2 对应区间（DIC 判据 + 力学峰值）。"""
    pre = [a for a in acc if a[1] < b["t_peak"]]
    post = [a for a in acc if a[0] >= b["t_peak"]]
    a1 = pre[0] if pre else [np.nan, np.nan]
    st = [("a 压密/早期", 0, 60),
          ("b 局部化前弹性", 60, t_ini),
          ("c 起始与缓慢演化", t_ini, a1[0]),
          ("d 第一次加速", a1[0], a1[1] + 1)]
    for i, a in enumerate(pre[1:], 2):
        st.append((f"c{i} 加速间缓慢段", st[-1][2], a[0]))
        st.append((f"d{i} 第 {i} 次加速", a[0], a[1] + 1))
    st.append(("峰前末段", st[-1][2], b["t_peak"]))
    if post:
        st.append(("e 峰后、再次加速前", b["t_peak"], post[0][0]))
        st.append(("f 峰后加速", post[0][0], post[0][1] + 1))
    return st


def h1(m, t_ini, acc):
    """局部化起始前后 AE 率。DIC 的 t_ini 不稳（268–362 s，见 methodScripts README），
    因此同时报告 t_ini 灵敏度全集上的结果，以及“DIC 速率首次明显抬升”= 第一次加速区间之前 10 s。"""
    sens = pd.read_csv(MS / "outputs" / "ren_tini_sensitivity.csv")
    rows = []
    for tt in sorted(set([t_ini, *sens.t_ini.dropna().round(0)])):
        before, after = mean_in(m, "hits", tt - 20, tt), mean_in(m, "hits", tt, tt + 20)
        rows.append(dict(t_ini=tt, ae_before=before, ae_after=after, ratio=after / before if before else np.nan))
    df = pd.DataFrame(rows)
    n_drop = int((df.ratio < 0.7).sum())
    verdict = "不支持" if n_drop == 0 else ("部分支持" if n_drop < len(df) / 2 else "支持")
    return df, dict(hypothesis="H1 局部化起始→AE 率骤降",
                    evidence=f"{len(df)} 个候选 t_ini 中 {n_drop} 个满足后 20 s/前 20 s AE 率 < 0.7；"
                             f"主口径 t_ini={t_ini:.0f}s: {df.loc[(df.t_ini - t_ini).abs().idxmin(), 'ratio']:.2f}",
                    verdict=verdict)


def h2(m, acc, b):
    rows = []
    for a0, a1 in acc:
        w = a1 + 1 - a0
        for k in NORMS:
            rows.append(dict(interval=f"{a0:.0f}-{a1 + 1:.0f}", norm=k,
                             ae_prev=mean_in(m, "hits", a0 - 20, a0), ae_in=mean_in(m, "hits", a0, a1 + 1 + 5),
                             shear_prev=shear_frac(m, a0 - 20, a0, k), shear_in=shear_frac(m, a0, a1 + 1 + 5, k)))
    df = pd.DataFrame(rows)
    main = df[(df.norm == NORM)]
    pre = main[[float(s.split("-")[0]) < b["t_peak"] for s in main.interval]]
    up_rate = int((pre.ae_in > 1.3 * pre.ae_prev).sum())
    up_sh = pd.Series({iv: float((g.shear_in > g.shear_prev).mean()) for iv, g in df.groupby("interval")})
    ev = (f"峰前 {len(pre)} 个加速区间中 AE 率升高(>1.3×) {up_rate} 个；"
          f"剪切占比升高比例（三种归一化平均）: " + ", ".join(f"{k}:{v:.2f}" for k, v in up_sh.items()))
    verdict = "支持" if up_rate == len(pre) and (up_sh > 0.5).all() else ("部分支持" if up_rate or (up_sh > 0.5).any() else "不支持")
    return df, dict(hypothesis="H2 加速演化→AE 率激增、剪切占比升高", evidence=ev, verdict=verdict)


def h3(m, quiet):
    """平静期内 DIC 相对位移是否继续增长；是否在平静期末开始加速；断面是否不同步。"""
    rows = []
    base = m[(m.sec >= 5) & (m.sec <= 60)]
    for q0, q1 in quiet["A_zero_run"]:
        r = dict(quiet=f"{q0:.0f}-{q1:.0f}", length_s=q1 - q0)
        for j in SECS:
            for k in ("dn_net", "ds_net"):
                c = f"{k}_{j}"
                x0 = m.loc[m.sec == q0, c].iloc[0]
                x1 = m.loc[m.sec == q1 - 1, c].iloc[0]
                noise = base[c].std()
                r[f"Δ{k}_{j}_um"] = (x1 - x0) * 1e3
                r[f"Δ{k}_{j}_snr"] = (x1 - x0) / noise
        q = m[(m.sec >= q0) & (m.sec < q1)]
        half = len(q) // 2
        r["rate_first_half_um_s"] = q.dmag__rate.iloc[:half].mean()
        r["rate_second_half_um_s"] = q.dmag__rate.iloc[half:].mean()
        r["rate_prev20_um_s"] = mean_in(m, "dmag__rate", q0 - 20, q0)
        r["rate_after10_um_s"] = mean_in(m, "dmag__rate", q1, q1 + 10)
        per = [q[f"dmag_{j}__rate"].mean() for j in SECS]
        r["section_rate_cv"] = np.std(per) / np.mean(per)
        r["section_rate_um_s"] = json.dumps([round(p, 2) for p in per])
        # 平静期前后 γmax 场幅值（原文图 14：0.025 → 0.043）
        r["gmax_p95_before"] = mean_in(m, "gmax_p95", q0 - 3, q0)
        r["gmax_p95_after"] = mean_in(m, "gmax_p95", q1, q1 + 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, dict(hypothesis="H3 AE 平静期内局部化继续发展", evidence="未识别到平静期", verdict="无法检验")
    r = df.iloc[0]
    grow = all(r[f"Δ{k}_{j}_snr"] > 3 for j in SECS for k in ("dn_net",)) or all(
        abs(r[f"Δ{k}_{j}_snr"]) > 3 for j in SECS for k in ("ds_net",))
    accel = r.rate_second_half_um_s > 1.5 * r.rate_first_half_um_s and r.rate_first_half_um_s > r.rate_prev20_um_s
    ev = (f"平静期 {r.quiet}s：δn 增量 " + "/".join(f"{r[f'Δdn_net_{j}_um']:.0f}" for j in SECS) + " μm，"
          f"δs 增量 " + "/".join(f"{r[f'Δds_net_{j}_um']:.0f}" for j in SECS) + " μm（基线噪声 ~0.5 μm）；"
          f"|δ| 速率 前20s {r.rate_prev20_um_s:.1f} → 平静期前半 {r.rate_first_half_um_s:.1f} → 后半 {r.rate_second_half_um_s:.1f}"
          f" → 结束后10s {r.rate_after10_um_s:.1f} μm/s；断面速率 CV={r.section_rate_cv:.2f}")
    verdict = "支持" if grow and accel else ("部分支持" if grow else "不支持")
    return df, dict(hypothesis="H3 AE 平静期内局部化继续发展、末段加速", evidence=ev, verdict=verdict)


def h4(m, t_ini, acc, quiet, b):
    pre = m[m.sec < b["t_peak"]]
    a1 = [a for a in acc if a[1] < b["t_peak"]][0]
    rows = []
    t_ae_peak = int(pre.sec[pre.rate5.idxmax()])
    for j in SECS:
        tU = int(pre.sec[pre[f"U_{j}"].idxmax()])
        tT = int(pre.sec[pre[f"tau_{j}"].abs().idxmax()])
        q0, q1 = quiet["A_zero_run"][0] if quiet["A_zero_run"] else (np.nan, np.nan)
        tau_q = m[(m.sec >= q0) & (m.sec < q1)][f"tau_{j}"]
        tau_after = mean_in(m, f"tau_{j}", q1, q1 + 5)
        rows.append(dict(section=j, t_U_peak=tU, t_tau_peak=tT, t_AE_rate5_peak=t_ae_peak,
                         U_peak_between_ini_and_acc=bool(t_ini <= tU <= a1[0]),
                         U_before_tau=bool(tU <= tT),
                         tau_rel_change_in_quiet=float((tau_q.iloc[-1] - tau_q.iloc[0]) / abs(tau_q.mean())),
                         tau_rel_change_after_quiet=float((tau_after - tau_q.iloc[-1]) / abs(tau_q.mean()))))
    df = pd.DataFrame(rows)
    ev = (f"AE 5 s 率峰值 {t_ae_peak}s；U 峰值 " + "/".join(str(x) for x in df.t_U_peak) +
          f"s；τ 峰值 " + "/".join(str(x) for x in df.t_tau_peak) + f"s；t_ini={t_ini:.0f}s，首次加速 {a1[0]:.0f}s；"
          f"平静期内 τ 相对变化 " + "/".join(f"{x:+.2f}" for x in df.tau_rel_change_in_quiet) +
          "，结束后 5 s " + "/".join(f"{x:+.2f}" for x in df.tau_rel_change_after_quiet))
    n_order = int(df.U_before_tau.sum())
    n_between = int(df.U_peak_between_ini_and_acc.sum())
    drop_after = int((df.tau_rel_change_after_quiet < -0.05).sum())
    verdict = ("部分支持" if (n_order >= 2) else "不支持") if drop_after == 0 else "支持"
    return df, dict(hypothesis="H4 U 峰≈AE 峰、位于起始与加速之间；平静期后 τ 陡降；能量先于应力释放",
                    evidence=ev + f"；U 峰在起始–加速之间的断面数 {n_between}/3，U 峰先于 τ 峰 {n_order}/3",
                    verdict=verdict)


def xcorr(m, b, max_lag=20):
    """DIC 速率与 AE 率的滞后 Spearman 相关（峰前），lag>0 表示 DIC 领先 AE。"""
    pre = m[(m.sec >= 60) & (m.sec < b["t_peak"])].reset_index(drop=True)
    rows = []
    for dic in ("dmag__rate", "dn_mean__rate", "ds_mean__rate", "gmax_lift", "crack_ratio"):
        x = pre[dic].diff() if dic in ("gmax_lift", "crack_ratio") else pre[dic]
        for ae in ("hits", "rate5", f"shear5_{NORM}"):
            for lag in range(-max_lag, max_lag + 1):
                y = pre[ae].shift(-lag)
                ok = x.notna() & y.notna()
                rows.append(dict(dic=dic, ae=ae, lag_s=lag, rho=spearmanr(x[ok], y[ok])[0]))
    return pd.DataFrame(rows)


def xcorr_detrended(m, b, max_lag=20, win=31):
    """去掉 31 s 滑动中位数趋势后再做滞后相关：两类信号都随加载单调增长（与时间 ρ≈0.9），
    原始相关主要反映共同趋势；去趋势后才是“事件级”对应。"""
    pre = m[(m.sec >= 60) & (m.sec < b["t_peak"])].reset_index(drop=True)
    det = lambda x: x - x.rolling(win, center=True, min_periods=10).median()
    rows = []
    for dic in ("dmag__rate", "dn_mean__rate", "ds_mean__rate"):
        x = det(pre[dic])
        for ae in ("rate5", f"shear5_{NORM}"):
            y0 = det(pre[ae])
            for lag in range(-max_lag, max_lag + 1):
                y = y0.shift(-lag)
                ok = x.notna() & y.notna()
                rows.append(dict(dic=dic, ae=ae, lag_s=lag, rho=spearmanr(x[ok], y[ok])[0]))
    return pd.DataFrame(rows)


def batching_diagnostic(m):
    """AE 时间戳是否“批量写入”：活跃段（5 s 率 > 50）出现整秒为 0 的格，且 0 之后紧跟的格计数偏高。"""
    act = m[m.rate5 > 50].reset_index(drop=True)
    z = act.hits == 0
    after = act.hits[z.shift(1, fill_value=False)]
    q = m.set_index("sec").hits
    return dict(active_secs=len(act), zero_secs_in_active=int(z.sum()),
                hits_after_zero_mean=float(after.mean()), hits_active_mean=float(act.hits.mean()),
                lag1_autocorr_active=float(act.hits.autocorr(1)),
                hits_356=float(q.get(356, np.nan)), hits_350_355_per_s=float(q.loc[350:355].mean()),
                hits_365_374_per_s=float(q.loc[365:374].mean()))


def plot(m, t_ini, acc, quiet, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    S = ["#2a78d6", "#eb6834", "#1baf7a"]
    p = m[(m.sec >= 250) & (m.sec <= 457)]
    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    axs[0].plot(p.sec, p.load_kN, color=ink, lw=1.2)
    axs[0].set_ylabel("载荷 (kN)")
    axs[1].bar(p.sec, p.hits, width=0.9, color="#c3c2b7", label="逐秒撞击数")
    axs[1].plot(p.sec, p.rate5, color=S[0], lw=1.4, label="5 s 移动平均撞击率")
    axs[1].set_ylim(0, 350)
    axs[1].text(356.5, 330, "356 s: 1128 次", fontsize=8, color=ink)
    axs[1].set_ylabel("AE 撞击 (次/s)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    for k, c in zip(NORMS, S):
        axs[2].plot(p.sec, p[f"shear_frac5_{k}"], color=c, lw=1.1, label=f"剪切占比 ({k} 归一化)")
    axs[2].set_ylabel("剪切型占比")
    axs[2].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    for j, c in zip(SECS, S):
        axs[3].plot(p.sec, p[f"dmag_{j}__rate"], color=c, lw=1.1, label=f"断面{j}")
    axs[3].set_ylim(-5, 60)
    axs[3].set_ylabel("带内外相对位移速率\n(μm/s, 5 s 窗)")
    axs[3].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    for j, c in zip(SECS, S):
        axs[4].plot(p.sec, p[f"U_{j}"] / p[f"U_{j}"].max(), color=c, lw=1.1, label=f"U 断面{j}")
        axs[4].plot(p.sec, p[f"tau_{j}"] / p[f"tau_{j}"].abs().max(), color=c, lw=1.1, ls="--", label=f"τ 断面{j}")
    axs[4].set_ylabel("归一化 U(实线) / τ(虚线)")
    axs[4].legend(frameon=False, fontsize=7, ncol=6, loc="lower left")
    axs[4].set_xlabel("力学时钟时间 (s)")
    for ax in axs:
        for a0, a1 in acc:
            ax.axvspan(a0, a1 + 1, color="#eb6834", alpha=0.10, lw=0)
        for q0, q1 in quiet["A_zero_run"]:
            ax.axvspan(q0, q1, color="#2a78d6", alpha=0.12, lw=0)
        for x in (b["t_ci"], b["t_cd"], b["t_peak"]):
            ax.axvline(x, color=muted, lw=0.8, ls=":")
        ax.axvline(t_ini, color=S[2], lw=1.0)
        ax.grid(axis="y", color=grid, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    fig.suptitle("Ren 式检验：AE 与 DIC 局部化（蓝底：AE 零计数段；橙底：DIC 加速区间；绿线：t_ini(5 s)；点线：起裂/损伤/峰值）",
                 fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ren_checks.png", dpi=110)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 200)
    m, t_ini, acc, quiet = load()
    b = C.stage_boundaries()

    st = ren_stages(t_ini, acc, b)
    tab = []
    for name, a0, a1 in st:
        r = dict(stage=name, t0=a0, t1=a1, ae_hits_per_s=mean_in(m, "hits", a0, a1),
                 dic_rate_um_s=mean_in(m, "dmag__rate", a0, a1))
        for k in NORMS:
            r[f"shear_frac_{k}"] = shear_frac(m, a0, a1, k)
        tab.append(r)
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "ren_stage_table.csv", index=False, encoding="utf-8-sig")
    print("== Ren 式六段划分（DIC 判据定义） ==")
    print(tab.round(3).to_string(index=False))

    res = []
    d1, r1 = h1(m, t_ini, acc); d1.to_csv(OUT / "h1_tini_ae.csv", index=False, encoding="utf-8-sig"); res.append(r1)
    d2, r2 = h2(m, acc, b); d2.to_csv(OUT / "h2_acc_ae.csv", index=False, encoding="utf-8-sig"); res.append(r2)
    d3, r3 = h3(m, quiet); d3.to_csv(OUT / "h3_quiet_dic.csv", index=False, encoding="utf-8-sig"); res.append(r3)
    d4, r4 = h4(m, t_ini, acc, quiet, b); d4.to_csv(OUT / "h4_energy_stress.csv", index=False, encoding="utf-8-sig"); res.append(r4)
    xc = xcorr(m, b)
    xc.to_csv(OUT / "xcorr.csv", index=False, encoding="utf-8-sig")
    best = xc.loc[xc.groupby(["dic", "ae"]).rho.idxmax()]
    z = xc[xc.lag_s == 0].set_index(["dic", "ae"]).rho
    best = best.assign(rho_lag0=[z[(r.dic, r.ae)] for r in best.itertuples()])
    tops = best.sort_values("rho", ascending=False).head(3)
    res.append(dict(hypothesis="整体：DIC 局部化演化与 AE 率对应良好",
                    evidence="峰前(60 s–峰值) Spearman 最大: " + "; ".join(
                        f"{r.dic}~{r.ae} ρ={r.rho:.2f}@lag{r.lag_s:+d}s (lag0 {r.rho_lag0:.2f})" for r in tops.itertuples()),
                    verdict="见证据"))
    xd = xcorr_detrended(m, b)
    xd.to_csv(OUT / "xcorr_detrended.csv", index=False, encoding="utf-8-sig")
    bd = xd.loc[xd.groupby(["dic", "ae"]).rho.idxmax()]
    z0 = xd[xd.lag_s == 0].set_index(["dic", "ae"]).rho
    res.append(dict(hypothesis="整体（去趋势）：事件级对应",
                    evidence="去 31 s 趋势后: " + "; ".join(
                        f"{r.dic}~{r.ae} 最大 ρ={r.rho:.2f}@lag{r.lag_s:+d}s (lag0 {z0[(r.dic, r.ae)]:.2f})"
                        for r in bd.itertuples()) + "（lag<0 表示 AE 领先 DIC）",
                    verdict="弱"))
    bt = batching_diagnostic(m)
    res.append(dict(hypothesis="数据诊断：AE 时间戳批量写入",
                    evidence=json.dumps({k: round(v, 2) for k, v in bt.items()}, ensure_ascii=False),
                    verdict="存在迹象" if bt["zero_secs_in_active"] > 5 and bt["hits_after_zero_mean"] > bt["hits_active_mean"] else "未见"))
    plot(m, t_ini, acc, quiet, b)
    hc = pd.DataFrame(res)
    hc.to_csv(OUT / "hypothesis_checks.csv", index=False, encoding="utf-8-sig")
    print("\n== H1 ==\n", d1.round(2).to_string(index=False))
    print("\n== H2 ==\n", d2[d2.norm == NORM].round(3).to_string(index=False))
    print("\n== H3 ==\n", d3.T.to_string(header=False))
    print("\n== H4 ==\n", d4.round(3).to_string(index=False))
    print("\n== 滞后互相关（最大 ρ） ==\n", best.round(3).to_string(index=False))
    print("\n== 结论汇总 ==")
    for r in res:
        print(f"[{r['verdict']}] {r['hypothesis']}\n     {r['evidence']}")


if __name__ == "__main__":
    main()
