"""02 逐帧 DIC 特征（以 Ren et al. 2026 的局部化方法为主，辅以场统计）。

Ren 方法在本数据上的实现
  1) γmax = (e1−e2)/2 = sqrt(((exx−eyy)/2)^2 + exy^2)   （exy 为张量剪应变；数据中 gamma 列是主方向角，不用）
  2) 参考帧（峰前 REF_LEAD_S 秒，带已清晰、尚未失配）上识别局部化带：γmax 鲁棒阈值 → 最大连通域 → PCA 主轴
  3) 沿带 25/50/75% 处设 3 个监测断面；每个断面在带两侧各取 3×3 网格点窗（≈原文 5×5 px），
     窗心距带边界 5 个网格点（≈原文 10 px）；窗固定在参考坐标（物质点）上，对所有帧使用
  4) 两窗中位位移差分解为法向（相对张拉，张开为正）和切向（相对滑移）分量；另给出扣除整帧
     均匀变形（仿射拟合）后的“净”分量——窗距 ~6 mm 时均匀应变本身就贡献同量级的相对位移
  5) 围岩应变能密度 U 与带上剪应力 τα（E=1, ν=0.25，只看形状）

逐帧还输出整体变形、γmax 分布、成带形态（逐帧自身识别，不用未来信息）等特征。
速率类特征不在这里做——单帧差分噪声大，放在 03 中与滑动窗口斜率一起比较。

输出 outputs/frame_features.csv, outputs/band_reference.json, outputs/fig_band_reference.png
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import common as C

SIGMA_MAX = 0.02          # 相关置信区间上限（全程 P99≈0.007）
EDGE_ROWS, EDGE_COLS = 5, 2   # 剔除压头区/侧边（网格点）
THR_IQR = 3.0             # 成带掩膜：γmax_s > P50 + THR_IQR·IQR（逐帧自归一）
MIN_CC_MM2 = 1.0
REF_LEAD_S = 5.0          # 参考帧 = 峰值前 5 s
SECTIONS = (0.25, 0.50, 0.75)
WIN_HALF = 1              # 3×3
WIN_GAP = 5               # 窗心距带边界的网格点数（10 px）
ENERGY_GAP = 15           # 能量区距带边界（30 px）
ENERGY_HALF = 3           # 7×7 网格点（≈15×15 px）
E_MOD, NU = 1.0, 0.25

XX, YY = C.grid_xy_mm()


def valid_mask(sig):
    m = np.isfinite(sig) & (sig <= SIGMA_MAX)
    m[:EDGE_ROWS] = m[-EDGE_ROWS:] = False
    m[:, :EDGE_COLS] = m[:, -EDGE_COLS:] = False
    return m


def gamma_max(exx, eyy, exy):
    return np.sqrt(((exx - eyy) / 2) ** 2 + exy ** 2)


def smooth_nan(a, size=3):
    """忽略 NaN 的均值滤波。"""
    m = np.isfinite(a)
    num = ndi.uniform_filter(np.where(m, a, 0.0), size)
    den = ndi.uniform_filter(m.astype(float), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m, num / den, np.nan)


def interior():
    m = np.zeros((C.NY, C.NX), bool)
    m[EDGE_ROWS:-EDGE_ROWS, EDGE_COLS:-EDGE_COLS] = True
    return m


INTERIOR = interior()


def cracked_mask(sig, valid0):
    """裂缝/失配点：参考状态可测、当前帧缺失或相关置信差（sigma>SIGMA_MAX）的内部点。"""
    return INTERIOR & valid0 & ~(np.isfinite(sig) & (sig <= SIGMA_MAX))


def band_mask(g, valid, cracked):
    """局部化掩膜 = γmax 鲁棒超限点 ∪ 裂缝点（裂缝处 DIC 已无法给出应变，但显然属于局部化带）。"""
    gs = smooth_nan(np.where(valid, g, np.nan))
    v = gs[valid]
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    m = valid & (gs > p50 + THR_IQR * (p75 - p25))
    m = ndi.binary_opening(m, np.ones((3, 3))) | cracked
    m = ndi.binary_closing(m, np.ones((3, 3))) & INTERIOR
    return m, gs


def morphology(m, gs, valid):
    """成带形态：超限面积占比、最大连通域占比、长短轴比、倾角、带宽、带内外对比、连通域数。"""
    nan = dict(A_loc=0.0, C_band=np.nan, LW=np.nan, angle_deg=np.nan, width_mm=np.nan, rho_in_out=np.nan, N_cc=0)
    lab, n = ndi.label(m, np.ones((3, 3)))
    if n == 0:
        return nan, None
    sizes = ndi.sum(m, lab, np.arange(1, n + 1))
    big = np.argmax(sizes) + 1
    lcc = lab == big
    x, y = XX[lcc], YY[lcc]
    P = np.c_[x - x.mean(), y - y.mean()]
    w, V = np.linalg.eigh(np.cov(P.T))
    major = V[:, 1] if V[1, 1] >= 0 else -V[:, 1]
    proj = P @ V
    L = proj[:, 1].max() - proj[:, 1].min() + C.H_MM
    area = lcc.sum() * C.CELL_AREA
    ring = ndi.binary_dilation(lcc, iterations=4) & ~ndi.binary_dilation(lcc, iterations=2) & valid
    out = dict(
        A_loc=m.sum() / valid.sum(),
        C_band=lcc.sum() / m.sum(),
        LW=np.sqrt(w[1] / max(w[0], 1e-12)),
        angle_deg=np.degrees(np.arctan2(major[0], major[1])),     # 与加载方向(y)夹角，右倾为正
        width_mm=area / L,
        rho_in_out=np.nanmedian(gs[lcc]) / np.nanmedian(gs[ring]) if ring.any() else np.nan,
        N_cc=int((sizes * C.CELL_AREA >= MIN_CC_MM2).sum()),
    )
    return out, lcc


def nearest_idx(x, y):
    """物理坐标 → 网格行列。"""
    col = int(round((x - XX[0, 0]) / (XX[0, 1] - XX[0, 0])))
    row = int(round((y - YY[0, 0]) / (YY[1, 0] - YY[0, 0])))
    return row, col


def build_reference(a, frame_ref, valid0):
    """参考带：参考帧掩膜的最大连通域；只取其“单分支段”（每行只有一段连续占据，自上而下最长连续行段），
    相当于 Ren 文中人工选定的主带。逐行取占据段中点得中心线，再在 25/50/75% 处设断面。"""
    exx, eyy, exy, sig = (np.array(a[f][frame_ref]) for f in ("exx", "eyy", "exy", "sigma"))
    valid = valid_mask(sig)
    cr = cracked_mask(sig, valid0)
    m, gs = band_mask(gamma_max(exx, eyy, exy), valid, cr)
    _, lcc = morphology(m, gs, valid)
    rows = []
    for r in range(C.NY):
        lab, n = ndi.label(lcc[r])
        rows.append((r, n, np.where(lab == 1)[0] if n == 1 else None))
    best, cur = [], []
    for r, n, cols in rows:
        if n == 1:
            cur.append((r, cols))
        else:
            best, cur = (cur if len(cur) > len(best) else best), []
    best = cur if len(cur) > len(best) else best
    rr = np.array([r for r, _ in best])
    cc = np.array([cols.mean() for _, cols in best])
    xs, ys = np.interp(cc, np.arange(C.NX), XX[0]), YY[rr, 0]

    def window_ok(rc, h):
        r, c = rc
        if not (h <= r < C.NY - h and h <= c < C.NX - h):
            return False
        return valid[r - h:r + h + 1, c - h:c + h + 1].mean() >= 5 / 9

    def section_at(i0):
        loc = np.abs(ys - ys[i0]) <= 2.0                             # ±2 mm 线性拟合求局部走向
        p = np.polyfit(ys[loc], xs[loc], 1)
        t = np.array([p[0], 1.0]) / np.hypot(p[0], 1.0)              # 切向，指向上
        n = np.array([t[1], -t[0]])                                   # 法向，t 顺时针 90°（指向右）
        c = np.array([np.polyval(p, ys[i0]), ys[i0]])

        def edge(sign):                                               # 沿法向走出 lcc 的距离
            for d in np.arange(0, 15, C.H_MM / 2):
                r_, c_ = nearest_idx(*(c + sign * d * n))
                if not (0 <= r_ < C.NY and 0 <= c_ < C.NX) or not lcc[r_, c_]:
                    return d
            return 15.0
        hR, hL = edge(+1), edge(-1)
        cen = c + (hR - hL) / 2 * n
        half = (hR + hL) / 2
        off, eoff = half + WIN_GAP * C.H_MM, half + ENERGY_GAP * C.H_MM
        s = dict(center_mm=cen.tolist(), t=t.tolist(), n=n.tolist(), half_width_mm=float(half),
                 angle_deg=float(np.degrees(np.arctan2(t[0], t[1]))),
                 win_R=nearest_idx(*(cen + off * n)), win_L=nearest_idx(*(cen - off * n)),
                 win_C=nearest_idx(*cen), win_E=nearest_idx(*(cen - eoff * n)))
        ok = window_ok(s["win_R"], WIN_HALF) and window_ok(s["win_L"], WIN_HALF)
        s["energy_ok"] = bool(window_ok(s["win_E"], ENERGY_HALF))
        return s, ok

    # 只在两侧窗都落在有效测点内的位置上布置断面（带靠近试样右缘，上段右窗会出界）
    cand = [(i, *section_at(i)) for i in range(2, len(rr) - 2)]
    feas = [(i, s) for i, s, ok in cand if ok]
    if len(feas) < len(SECTIONS):
        raise RuntimeError("参考带上可布置监测窗的位置不足")
    secs = []
    for q in SECTIONS:
        i, s = feas[int(round(q * (len(feas) - 1)))]
        s["q"] = q
        secs.append(s)
    return dict(frame=int(frame_ref), sections=secs, lcc_area_mm2=float(lcc.sum() * C.CELL_AREA),
                main_band_y_mm=[float(ys.min()), float(ys.max())],
                feasible_y_mm=[float(ys[feas[0][0]]), float(ys[feas[-1][0]])]), lcc, gs


def win_median(arr, rc, h):
    r, c = rc
    return np.nanmedian(arr[max(r - h, 0): r + h + 1, max(c - h, 0): c + h + 1])


def affine_grad(u, v, valid):
    """整帧仿射拟合 u,v = a + b x + c y → 2×2 均匀位移梯度。"""
    x, y = XX[valid], YY[valid]
    A = np.c_[np.ones_like(x), x, y]
    cu = np.linalg.lstsq(A, u[valid], rcond=None)[0]
    cv = np.linalg.lstsq(A, v[valid], rcond=None)[0]
    return np.array([[cu[1], cu[2]], [cv[1], cv[2]]])


def stresses(exx, eyy, exy):
    k = E_MOD / (1 - NU ** 2)
    return k * (exx + NU * eyy), k * (eyy + NU * exx), E_MOD / (1 + NU) * exy


def frame_features(a, k, ref, valid0):
    u, v, exx, eyy, exy, sig = (np.array(a[f][k]) for f in ("u_c", "v_c", "exx", "eyy", "exy", "sigma"))
    valid = valid_mask(sig)
    nv = valid.sum()
    cr = cracked_mask(sig, valid0)
    row = dict(frame=k, valid_ratio=nv / valid0.sum(), crack_ratio=cr.sum() / (INTERIOR & valid0).sum())
    if nv < 1000:
        return row
    g = gamma_max(exx, eyy, exy)
    gv, e1 = g[valid], ((exx + eyy) / 2 + g)[valid]
    p50, p95, p99 = np.percentile(gv, [50, 95, 99])
    row.update(
        exx_med=np.median(exx[valid]), eyy_med=np.median(eyy[valid]),
        ev2d_med=np.median((exx + eyy)[valid]),
        gmax_p50=p50, gmax_p95=p95, gmax_p99=p99, gmax_lift=p95 - p50, gmax_cv=gv.std() / gv.mean(),
        F_e1_005=(e1 >= 0.005).mean(),                               # Dong εmd≈0.005 的超限面积（辅助）
    )
    m, gs = band_mask(g, valid, cr)
    row.update(morphology(m, gs, valid)[0])

    G = affine_grad(u, v, valid)
    for j, s in enumerate(ref["sections"], 1):
        t, n = np.array(s["t"]), np.array(s["n"])
        dR = np.array([win_median(u, s["win_R"], WIN_HALF), win_median(v, s["win_R"], WIN_HALF)])
        dL = np.array([win_median(u, s["win_L"], WIN_HALF), win_median(v, s["win_L"], WIN_HALF)])
        sep = np.array([XX[tuple(s["win_R"])] - XX[tuple(s["win_L"])], YY[tuple(s["win_R"])] - YY[tuple(s["win_L"])]])
        rel = dR - dL
        net = rel - G @ sep
        row.update({f"dn_{j}": rel @ n, f"ds_{j}": rel @ t, f"dn_net_{j}": net @ n, f"ds_net_{j}": net @ t})
    s = ref["sections"][1]                                           # 中段断面
    alpha = np.radians(s["angle_deg"])
    sx, sy, txy = stresses(*(win_median(f, s["win_E"], ENERGY_HALF) for f in (exx, eyy, exy)))
    smean, srad = (sx + sy) / 2, np.hypot((sx - sy) / 2, txy)
    s1, s2 = smean + srad, smean - srad
    row["U_surr"] = (s1 ** 2 + s2 ** 2 - 2 * NU * s1 * s2) / (2 * E_MOD)
    sx, sy, txy = stresses(*(win_median(f, s["win_C"], WIN_HALF) for f in (exx, eyy, exy)))
    row["tau_band"] = 0.5 * (sx - sy) * np.sin(2 * alpha) + txy * np.cos(2 * alpha)
    return row


def plot_reference(a, ref, gs, fi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, axs = plt.subplots(1, 4, figsize=(11, 5.6), constrained_layout=True)
    for axx, tt in zip(axs, [300, 360, fi.loc[ref["frame"], "t_s"], 445]):
        k = int(np.argmin(np.abs(fi.t_s.to_numpy() - tt)))
        exx, eyy, exy, sig = (np.array(a[f][k]) for f in ("exx", "eyy", "exy", "sigma"))
        g = np.where(valid_mask(sig), gamma_max(exx, eyy, exy), np.nan)
        axx.imshow(g, cmap="magma", vmin=0, vmax=np.nanpercentile(g, 99.5), extent=ext, origin="upper")
        axx.set_title(f"t = {fi.t_s[k]:.0f} s  {fi.stage_zh[k]}", fontsize=10)
        axx.set_xlabel("x (mm)")
        if k == ref["frame"]:
            for s in ref["sections"]:
                for key, mk in (("win_L", "s"), ("win_R", "s"), ("win_E", "^")):
                    r, c = s[key]
                    axx.plot(XX[r, c], YY[r, c], mk, ms=6, mfc="none", mec="#2a78d6" if mk == "s" else "#1baf7a", mew=2)
                cx, cy = s["center_mm"]
                axx.text(cx + 1.5, cy, f"{int(s['q']*100)}%", color="white", fontsize=8)
    axs[0].set_ylabel("y (mm)")
    fig.suptitle("γmax 场与 Ren 式监测窗（方框：带两侧位移窗；三角：围岩能量区）", fontsize=11)
    fig.savefig(C.OUT / "fig_band_reference.png", dpi=110)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    a = C.open_cache()
    fi = pd.read_csv(C.OUT / "frame_index.csv")
    b = C.stage_boundaries()
    k_ref = int(np.argmin(np.abs(fi.t_s.to_numpy() - (b["t_peak"] - REF_LEAD_S))))
    valid0 = valid_mask(np.array(a["sigma"][1]))
    ref, _, gs = build_reference(a, k_ref, valid0)
    ref["t_s"] = float(fi.t_s[k_ref])
    (C.OUT / "band_reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    plot_reference(a, ref, gs, fi)

    rows = []
    for k in range(C.N_FRAMES):
        rows.append(frame_features(a, k, ref, valid0))
        if k % 200 == 0:
            print(f"\r帧 {k}/{C.N_FRAMES}", end="", flush=True)
    print()
    ff = pd.DataFrame(rows)
    ff = fi[["frame", "t_s", "load_kN", "stage_code", "stage", "uncertain"]].merge(ff, on="frame")
    ff.to_csv(C.OUT / "frame_features.csv", index=False, encoding="utf-8-sig")
    print("参考帧", k_ref, "t=%.1f s" % ref["t_s"])
    for s in ref["sections"]:
        print("  断面 %.0f%%: 中心 (%.1f, %.1f) mm  倾角 %.1f°  半宽 %.2f mm" %
              (s["q"] * 100, *s["center_mm"], s["angle_deg"], s["half_width_mm"]))
    print("特征列:", [c for c in ff.columns if c not in fi.columns])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
