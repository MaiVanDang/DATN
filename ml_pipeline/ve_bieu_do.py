#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ve_bieu_do.py — Sinh toàn bộ biểu đồ minh họa dữ liệu cho ĐATN BioAuth.

Gộp 3 nhóm biểu đồ vào một script, dùng chung hàm với step2_preprocess.py:

  A. Dữ liệu SAU tiền xử lý  (đọc processed/<user>/*.npy)
       phanbo_theo_cauhinh.png   số cửa sổ theo cấu hình (walking vs all)
       phanbo_theo_nguoi.png     số cửa sổ theo người dùng

  B. Tín hiệu THÔ            (đọc data/<user>/<session>/*.csv)
       tin_hieu_di_bo.png        gia tốc kế thô khi đi bộ (động)
       tin_hieu_ngoi.png         gia tốc kế thô khi ngồi  (tĩnh)

  C. Minh họa cửa sổ trượt & chuẩn hóa Z-score (đọc data/*.csv) — hình đơn, không tiêu đề
       cua_so_truot.png                               cửa sổ trượt 200 mẫu, bước 20
       2a_zscore_truoc.png / 2b_zscore_sau.png        Z-score giữa người dùng
       3a_scale_truoc.png / 3b_scale_sau.png          Z-score scale 9 kênh
       4a_window_truoc.png / 4b_window_zscore_sau.png một cửa sổ 4s

Chạy:
    python ve_bieu_do.py --data_dir ./data --proc_dir ./processed --out ./plots
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import step2_preprocess as s

USER_DON = "user16"

HZ       = s.HZ
ACC      = ["acc_x", "acc_y", "acc_z"]
CHANNELS = s.SENSOR_COLS
ACC_COL  = {"acc_x": "#4C72B0", "acc_y": "#C44E52", "acc_z": "#55A868"}
USER_COL = ["#378ADD", "#E07020", "#9B59B6", "#1D9E75"]
C_BEFORE = "#CC3333"
C_AFTER  = "#1D9E75"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, out, name):
    path = Path(out) / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {name}")


def _walking_files(data_dir):
    """[(user, file walking đầu tiên)] cho mỗi user."""
    out = {}
    for udir in sorted(Path(data_dir).iterdir()):
        if udir.is_dir():
            for f in sorted(udir.rglob("walking_*.csv")):
                out.setdefault(udir.name, f)
                break
    return list(out.items())


def _first_segment(path):
    df = pd.read_csv(path).dropna(subset=CHANNELS)
    ts, to = s._ts_info(df)
    segs = s.split_segments(df)
    return (segs[0], ts, to) if segs else (None, ts, to)


def _pick_raw(data_dir, activity, user, session):
    """File <activity>_*.csv đầu tiên khớp user + session."""
    pat = re.compile(rf"session[_]?0*{session}$", re.I)
    for f in sorted(glob.glob(str(Path(data_dir) / "**" / f"{activity}_*.csv"),
                              recursive=True)):
        parts = Path(f).parts
        if (not user or user in parts) and any(pat.match(p) for p in parts):
            return f
    return None


def _load_processed(proc_dir):
    """Đọc .npy theo từng user -> dict {cfg: {X, y}} cho 'all' và 'walking'."""
    res = {"all": {"X": [], "y": []}, "walking": {"X": [], "y": []}}
    for d in sorted(p for p in Path(proc_dir).iterdir() if p.is_dir()):
        for cfg, xn, yn in [("all", "X_inertial.npy", "y_inertial.npy"),
                            ("walking", "X_walking.npy", "y_walking.npy")]:
            xp, yp = d / xn, d / yn
            if xp.exists() and yp.exists():
                X = np.load(xp)
                y = np.load(yp, allow_pickle=True)
                if len(X):
                    res[cfg]["X"].append(X)
                    res[cfg]["y"].append(np.asarray(y, dtype=object))
    for cfg in res:
        if res[cfg]["X"]:
            res[cfg]["X"] = np.concatenate(res[cfg]["X"])
            res[cfg]["y"] = np.concatenate(res[cfg]["y"])
        else:
            res[cfg]["X"] = np.empty((0, 200, 9))
            res[cfg]["y"] = np.array([])
    return res


def _user_of(y):
    """Nhãn '<uid>_session_k' -> '<uid>'."""
    return np.array([str(l).split("_session")[0] for l in y])


def _unum(u):
    m = re.search(r"\d+", u)
    return int(m.group()) if m else 0


def plot_phanbo(res, out):
    n_all, n_walk = len(res["all"]["y"]), len(res["walking"]["y"])
    users = _user_of(res["all"]["y"])
    uniq = sorted(set(users), key=_unum)
    counts = [int(np.sum(users == u)) for u in uniq]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(uniq)), counts, color="#4C72B0")
    ax.set_xlabel("Người dùng"); ax.set_ylabel("Số cửa sổ")
    ax.set_xticks(range(len(uniq)))
    ax.set_xticklabels([f"U{_unum(u)}" for u in uniq], rotation=90, fontsize=8)
    if counts:
        tb = float(np.mean(counts))
        ax.axhline(tb, ls="--", c="gray", lw=1)
        ax.text(len(uniq) - 1, tb, f" TB={tb:.0f}", va="bottom", ha="right",
                fontsize=9, color="gray")
    _save(fig, out, "phanbo_theo_nguoi.png")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["Walking", "All"], [n_walk, n_all], color=["#55A868", "#4C72B0"])
    ax.set_ylabel("Số cửa sổ")
    for i, v in enumerate([n_walk, n_all]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    _save(fig, out, "phanbo_theo_cauhinh.png")


def plot_tin_hieu_tho(data_dir, out, user, session, seconds):
    fw = _pick_raw(data_dir, "walking", user, session)
    fs = _pick_raw(data_dir, "sitting", user, session)
    if not fw or not fs:
        print(f"  [bỏ qua] thiếu walking/sitting (user={user}, session={session})")
        return

    n = int(round(seconds * HZ))
    di_bo = pd.read_csv(fw)[ACC].to_numpy(float)[:n]
    ngoi  = pd.read_csv(fs)[ACC].to_numpy(float)[:n]

    allv = np.concatenate([di_bo.ravel(), ngoi.ravel()])
    pad = 0.05 * (allv.max() - allv.min() + 1e-9)
    ylim = (allv.min() - pad, allv.max() + pad)

    for win, name in [(di_bo, "tin_hieu_di_bo.png"), (ngoi, "tin_hieu_ngoi.png")]:
        t = np.arange(len(win)) / HZ
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for k, ch in enumerate(ACC):
            ax.plot(t, win[:, k], label=ch, lw=1.0, color=ACC_COL[ch])
        ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("Gia tốc (m/s$^2$)")
        ax.set_xlim(0, t[-1] if len(t) > 1 else 1); ax.set_ylim(ylim)
        ax.legend(loc="upper right", fontsize=9)
        _save(fig, out, name)


def plot_zscore_users(files, out, n_users=3):
    picked = []
    for uid, f in files:
        seg, ts, to = _first_segment(f)
        if seg is None or len(seg) < s.WINDOW_SIZE:
            continue
        raw = seg["acc_x"].values[:s.WINDOW_SIZE].astype(np.float64)
        picked.append((uid, raw, (raw - raw.mean()) / (raw.std() + 1e-8)))
        if len(picked) == n_users:
            break
    if len(picked) < 2:
        print("  [bỏ qua] zscore_users: cần ≥2 user")
        return

    t = np.arange(s.WINDOW_SIZE) / s.HZ

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for (uid, raw, _z), c in zip(picked, USER_COL):
        ax.plot(t, raw, color=c, lw=1.1, alpha=0.85, label=uid)
    ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("acc_x (m/s$^2$)")
    ax.legend(loc="upper right", fontsize=9, title="Người dùng")
    _save(fig, out, "2a_zscore_truoc.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    for (uid, _raw, z), c in zip(picked, USER_COL):
        ax.plot(t, z, color=c, lw=1.1, alpha=0.85, label=uid)
    ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("acc_x (chuẩn hóa)")
    ax.legend(loc="upper right", fontsize=9, title="Người dùng")
    _save(fig, out, "2b_zscore_sau.png")


def plot_scale_kenh(files, out):
    raw_c = {c: [] for c in CHANNELS}
    z_c   = {c: [] for c in CHANNELS}
    for _, f in files:
        df = pd.read_csv(f).dropna(subset=CHANNELS)
        for seg in s.split_segments(df):
            data = seg[CHANNELS].values.astype(np.float64)
            for st in range(0, len(data) - s.WINDOW_SIZE + 1, s.WINDOW_SIZE):
                w = data[st:st + s.WINDOW_SIZE]
                wz = (w - w.mean(0)) / (w.std(0) + 1e-8)
                for j, c in enumerate(CHANNELS):
                    raw_c[c].extend(w[:, j][::10])
                    z_c[c].extend(wz[:, j][::10])
            break
        if all(len(v) > 2000 for v in raw_c.values()):
            break

    box_colors = ["#378ADD"] * 3 + ["#E07020"] * 3 + ["#9B59B6"] * 3
    for src, name in [(raw_c, "3a_scale_truoc.png"), (z_c, "3b_scale_sau.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bp = ax.boxplot([src[c] or [0] for c in CHANNELS], patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.3))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_xticks(range(1, 10))
        ax.set_xticklabels(CHANNELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Giá trị")
        _save(fig, out, name)


def plot_window_zscore(files, out):
    seg = None
    for uid, f in files:
        seg, ts, to = _first_segment(f)
        if seg is not None and len(seg) >= s.WINDOW_SIZE:
            break
    if seg is None:
        print("  [bỏ qua] window_zscore: không có cửa sổ")
        return

    cols = ["acc_x", "gyro_x", "mag_x"]
    w = seg[cols].values[:s.WINDOW_SIZE].astype(np.float64)
    wz = (w - w.mean(0)) / (w.std(0) + 1e-8)
    t = np.arange(s.WINDOW_SIZE) / s.HZ

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, (c, col) in enumerate(zip(cols, USER_COL)):
        ax.plot(t, w[:, j], color=col, lw=1.0, alpha=0.85, label=c)
    ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("Giá trị thô")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, out, "4a_window_truoc.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    for j, (c, col) in enumerate(zip(cols, USER_COL)):
        ax.plot(t, wz[:, j], color=col, lw=1.0, alpha=0.85, label=c)
    ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("Giá trị chuẩn hóa")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, out, "4b_window_zscore_sau.png")


def plot_cua_so_truot(files, out, n_win=6):
    """Minh họa cửa sổ trượt: tín hiệu liên tục được cắt thành các cửa sổ
    WINDOW_SIZE mẫu, trượt với bước STRIDE (chồng lấn cao)."""
    seg = None
    for _, f in files:
        seg, _ts, _to = _first_segment(f)
        if seg is not None:
            break
    if seg is None:
        print("  [bỏ qua] cửa sổ trượt: không có đoạn")
        return

    win_sec = s.WINDOW_SIZE / HZ
    stride_sec = s.STRIDE / HZ
    span_sec = win_sec + (n_win - 1) * stride_sec
    n = min(len(seg), int(span_sec * HZ) + 1)
    sig = seg["acc_x"].values[:n].astype(np.float64)
    t = np.arange(n) / HZ

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, sig, color="#4C72B0", lw=1.0, zorder=3)

    y0, y1 = float(sig.min()), float(sig.max())
    yr = (y1 - y0) or 1.0
    base = y0 - 0.20 * yr
    dy = 0.13 * yr
    h = 0.06 * yr
    colors = ["#E07020", "#1D9E75", "#9B59B6", "#C44E52", "#378ADD", "#E6C200"]
    for k in range(n_win):
        x0 = k * stride_sec
        yk = base - k * dy
        ax.add_patch(Rectangle((x0, yk - h / 2), win_sec, h,
                               facecolor=colors[k % len(colors)], alpha=0.6,
                               edgecolor="black", lw=0.5, zorder=2))
        ax.text(x0 + 0.05, yk, f"{k + 1}", va="center", fontsize=7.5, zorder=4)

    yw = base + h / 2 + 0.05 * yr
    ax.annotate("", xy=(win_sec, yw), xytext=(0, yw),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(win_sec / 2, yw + 0.02 * yr, f"Cửa sổ = {win_sec:.0f}s ({s.WINDOW_SIZE} mẫu)",
            ha="center", va="bottom", fontsize=8.5)
    y3 = base - dy - h / 2 - 0.05 * yr
    ax.annotate("", xy=(stride_sec, y3), xytext=(0, y3),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1))
    ax.text(stride_sec + 0.05, y3, f"bước {stride_sec:.1f}s ({s.STRIDE} mẫu)",
            ha="left", va="center", fontsize=8, color="red")

    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("acc_x (m/s$^2$)")
    ax.set_ylim(base - n_win * dy - 0.05 * yr, y1 + 0.15 * yr)
    _save(fig, out, "cua_so_truot.png")


def main():
    ap = argparse.ArgumentParser(description="Sinh toàn bộ biểu đồ minh họa dữ liệu")
    ap.add_argument("--data_dir", default="./data", help="Thư mục dữ liệu thô (.csv)")
    ap.add_argument("--proc_dir", default="./processed", help="Thư mục đã tiền xử lý (.npy)")
    ap.add_argument("--out", default="./plots", help="Thư mục lưu biểu đồ")
    ap.add_argument("--user", default=USER_DON, help="User cho các biểu đồ 1 user (1, 4, tín hiệu thô)")
    ap.add_argument("--session", default="1")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("[A] Phân bố sau tiền xử lý")
    if Path(args.proc_dir).is_dir():
        res = _load_processed(args.proc_dir)
        if len(res["all"]["y"]):
            plot_phanbo(res, out)
        else:
            print("  [bỏ qua] không đọc được .npy hợp lệ trong", args.proc_dir)
    else:
        print("  [bỏ qua] không thấy", args.proc_dir)

    print("[B] Tín hiệu thô (đi bộ vs ngồi)")
    plot_tin_hieu_tho(args.data_dir, out, args.user, args.session, args.seconds)

    print("[C] Minh họa cửa sổ trượt & chuẩn hóa Z-score")
    files = _walking_files(args.data_dir)
    if not files:
        print("  [bỏ qua] không thấy walking_*.csv trong", args.data_dir)
    else:
        one = [(u, f) for u, f in files if u == args.user]
        if not one:
            print(f"  [bỏ qua] biểu đồ 1 user: không thấy {args.user}")
        else:
            plot_cua_so_truot(one, out)
            plot_window_zscore(one, out)
        plot_zscore_users(files, out)
        plot_scale_kenh(files, out)

    print(f"\n✓ Hoàn tất. Biểu đồ tại: {out.resolve()}/")


if __name__ == "__main__":
    main()
