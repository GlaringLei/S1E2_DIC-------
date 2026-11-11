"""01 读入 DIC 逐帧点场 → 规则网格缓存；估计 DIC 帧时间偏移；把每帧对齐到人工阶段标注。

输出
  cache/{u_c,v_c,exx,eyy,exy,sigma}.npy   (2300, 245, 116) float32，缺失点 NaN
  outputs/frame_index.csv                 每帧: frame, t_s, load_kN, stage, stage_zh, uncertain, n_valid
  outputs/alignment.json                  t0、估计依据、阶段边界
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common as C

COLS = ["x", "y", "sigma", "u_c", "v_c", "exx", "eyy", "exy"]


def _read_frame(k: int):
    d = pd.read_csv(C.frame_path(k), skipinitialspace=True)
    d.columns = [c.strip().strip('"') for c in d.columns]
    d = d[COLS]
    ix = ((d.x.to_numpy() - C.X0) // C.STEP).astype(int)
    iy = ((d.y.to_numpy() - C.Y0) // C.STEP).astype(int)
    ok = d.sigma.to_numpy() >= 0                       # Vic-2D 失配点 sigma=-1（或整行删除）
    return k, iy[ok], ix[ok], {f: d[f].to_numpy(np.float32)[ok] for f in C.FIELDS}


def _worker(ks):
    arrs = C.open_cache("r+")
    for k in ks:
        _, iy, ix, vals = _read_frame(k)
        for f in C.FIELDS:
            arrs[f][k, iy, ix] = vals[f]
    for a in arrs.values():
        a.flush()
    return len(ks)


def build_cache(n_jobs=12):
    if all((C.CACHE / f"{f}.npy").exists() for f in C.FIELDS):
        print("cache 已存在，跳过读入")
        return
    for f in C.FIELDS:
        a = np.lib.format.open_memmap(C.CACHE / f"{f}.npy", mode="w+", dtype=np.float32,
                                      shape=(C.N_FRAMES, C.NY, C.NX))
        a[:] = np.nan
        a.flush()
        del a
    chunks = np.array_split(np.arange(C.N_FRAMES), n_jobs * 4)
    with ProcessPoolExecutor(n_jobs) as ex:
        done = 0
        for n in ex.map(_worker, chunks):
            done += n
            print(f"\r读入 {done}/{C.N_FRAMES}", end="", flush=True)
    print()


def frame_quick_stats():
    a = C.open_cache()
    n_valid = np.isfinite(a["eyy"]).sum(axis=(1, 2))
    eyy_med = np.nanmedian(a["eyy"].reshape(C.N_FRAMES, -1), axis=1)
    # 观测区上下两条带的平均竖向位移之差 → 虚拟引伸计（轴向缩短，mm），对噪声稳健
    v = a["v_c"]
    top = np.nanmedian(v[:, 10:30, 10:-10].reshape(C.N_FRAMES, -1), axis=1)
    bot = np.nanmedian(v[:, -30:-10, 10:-10].reshape(C.N_FRAMES, -1), axis=1)
    return n_valid, eyy_med, top - bot


def estimate_t0(ext, n_valid):
    """t0 = 第 0 帧在力学时钟上的时刻。

    帧率按已知 5 fps 固定，只估计偏移。依据：峰前小台阶与峰后载荷骤降时，虚拟引伸计同步突变。
    在 t0 ∈ [-10, 10] s、步长 0.02 s 上，最大化 Δload 与 Δext 的 Pearson 相关（都按 0.2 s 帧网格差分）。
    稳健处理：只用 1500 帧之后（峰前 ~130 s 起，信号强）、且排除最后的整体崩解帧（≥2284，单帧 −282 μm，
    会单独主导相关并把 t0 拉到 −8.8 s 的伪解）；Δext 按 0.5/99.5 分位截尾。
    """
    tm, load, _ = C.read_load()
    k = np.arange(C.N_FRAMES)
    dext = np.diff(ext)
    lo, hi = np.nanpercentile(dext[1500:2283], [0.5, 99.5])
    dext = np.clip(dext, lo, hi)
    use = (k[1:] >= 1500) & (k[1:] < 2284)
    best = []
    for t0 in np.arange(-10, 10.0001, 0.02):
        t = C.frame_time(k, t0)
        ok = (t >= tm[0]) & (t <= tm[-1])
        L = np.interp(t, tm, load)
        dL = np.diff(L)
        m = ok[1:] & ok[:-1] & np.isfinite(dext) & use
        if m.sum() < 500:
            continue
        r = np.corrcoef(dL[m], dext[m])[0, 1]
        best.append((t0, r))
    best = np.array(best)
    t0 = float(best[np.argmax(best[:, 1]), 0])
    # 佐证：峰值时刻 vs 第一次大规模失配的帧
    drop = np.diff(n_valid.astype(float))
    k_drop = int(np.argmin(drop[: int((tm[np.argmax(load)] + 20) * C.FPS)]) + 1)
    return t0, best, k_drop


def main():
    build_cache()
    n_valid, eyy_med, ext = frame_quick_stats()
    t0, scan, k_drop = estimate_t0(ext, n_valid)
    b = C.stage_boundaries()
    k = np.arange(C.N_FRAMES)
    t = C.frame_time(k, t0)
    tm, load, _ = C.read_load()
    code, unc = C.label_times(t, b)
    code[t > tm[-1]] = -2                                       # 超出力学记录
    stage = np.where(code == -2, "after_end", np.where(code == -1, "before_start", np.array(C.STAGE_ORDER, dtype=object)[np.clip(code, 0, 3)]))
    fi = pd.DataFrame({
        "frame": k, "t_s": t.round(3), "load_kN": np.interp(t, tm, load, left=np.nan, right=np.nan).round(4),
        "stage_code": code, "stage": stage, "stage_zh": [C.STAGE_ZH.get(s, s) for s in stage],
        "uncertain": unc.astype(int), "n_valid": n_valid, "eyy_median": eyy_med, "ext_mm": ext,
    })
    fi.to_csv(C.OUT / "frame_index.csv", index=False, encoding="utf-8-sig")

    top = scan[np.argsort(-scan[:, 1])[:5]]
    info = {
        "t0_s": t0,
        "rule": "t_s = t0 + frame/5；frame 为文件编号 HS-2-XXXX",
        "t0_method": "最大化 Δ载荷 与 Δ虚拟引伸计 的相关，扫描 t0∈[-10,10] s，步长 0.02 s",
        "t0_scan_top5": [[float(a), float(r)] for a, r in top],
        "check_first_mass_mismatch_frame": k_drop,
        "check_first_mass_mismatch_time_s": float(C.frame_time(k_drop, t0)),
        "load_peak_time_s": b["t_peak"],
        "boundaries_s": b,
        "label_clock": "标注表 mechanics_time_s = AE 到达时间 − 40 s，按整秒给出，边界精度 ±1 s",
        "frames_per_stage": fi.stage.value_counts().to_dict(),
        "frames_uncertain": int(unc.sum()),
    }
    (C.OUT / "alignment.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({kk: info[kk] for kk in ["t0_s", "t0_scan_top5", "check_first_mass_mismatch_time_s",
                                              "load_peak_time_s", "frames_per_stage", "frames_uncertain"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
