import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ─────────────────────────────────────
HZ          = 50
WINDOW_SIZE = 4 * HZ          # 200 samples = 4 giây
STRIDE      = 20              # 0.4 s / window → ~80% overlap
MAX_GAP_SEC = 5 / HZ          # gap > 0.1 s → cắt segment
BURST_THR   = 150             # ms: inter-key được xem là "burst"

SENSOR_COLS = [
    "acc_x","acc_y","acc_z",
    "gyro_x","gyro_y","gyro_z",
    "mag_x","mag_y","mag_z",
]

# ── PATH ────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
PROC_DIR = Path(__file__).parent / "processed"

# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════
def split_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Tách DataFrame IMU thành các đoạn liên tục, bỏ gap > MAX_GAP_SEC."""
    ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else "timestamp_ns"
    to_sec = 1e3 if ts_col == "timestamp_ms" else 1e9
    df = df.sort_values(ts_col).reset_index(drop=True)
    diffs = df[ts_col].diff() / to_sec
    gap_idx = list(diffs[diffs > MAX_GAP_SEC].index)
    bounds  = [0] + gap_idx + [len(df)]
    segs = []
    for i in range(len(bounds) - 1):
        seg = df.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True)
        if len(seg) >= WINDOW_SIZE:
            segs.append(seg)
    return segs

def zscore_windows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-window, per-channel z-score: mean=0, std=1 theo chiều time (axis=1)."""
    mean = X.mean(axis=1, keepdims=True)        # (N, 1, 9)
    std  = X.std(axis=1, keepdims=True) + eps   # (N, 1, 9)
    return (X - mean) / std

def make_windows(df: pd.DataFrame, label: str):
    """Sliding-window trên DataFrame IMU, trả về (X, y)."""
    df = df.dropna(subset=SENSOR_COLS)
    X_list = []
    for seg in split_segments(df):
        data = seg[SENSOR_COLS].values.astype(np.float64)
        for s in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
            X_list.append(data[s:s + WINDOW_SIZE])
    if not X_list:
        return np.empty((0, WINDOW_SIZE, len(SENSOR_COLS)), np.float64), np.array([])
    X = zscore_windows(np.array(X_list, np.float64))
    return X, np.full(len(X), label)

def pct5(arr):
    """Trả về (p25, median, p75) hoặc (nan, nan, nan) nếu rỗng."""
    if len(arr) < 2:
        return np.nan, np.nan, np.nan
    return float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75))

def stats5(arr, prefix: str) -> dict:
    """mean / std / median / p25 / p75."""
    p25, med, p75 = pct5(arr)
    m  = float(np.mean(arr))   if len(arr) >= 2 else np.nan
    sd = float(np.std(arr))    if len(arr) >= 2 else np.nan
    return {f"{prefix}_mean": m, f"{prefix}_std": sd,
            f"{prefix}_median": med, f"{prefix}_p25": p25, f"{prefix}_p75": p75}

# ═══════════════════════════════════════════════
# IMU
# ═══════════════════════════════════════════════
ACTIVITIES = ["walking", "sitting", "standing"]

def _read_activity(sess_dir: Path, activity: str) -> pd.DataFrame | None:
    dfs = []
    for f in sorted(sess_dir.glob(f"{activity}_att*.csv")):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"    [W] {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

def process_inertial(sess_dir: Path, label: str):
    """
    Trả về 2 tuple:
      (X_walking, y_walking)   — chỉ walking
      (X_inertial, y_inertial) — walking + sitting + standing
    """
    X_walk, y_walk = [], []
    X_all,  y_all  = [], []
    empty = np.empty((0, WINDOW_SIZE, len(SENSOR_COLS)), np.float64)

    for act in ACTIVITIES:
        df = _read_activity(sess_dir, act)
        if df is None:
            continue
        X, y = make_windows(df, label)
        if len(X) == 0:
            continue
        X_all.append(X);  y_all.append(y)
        if act == "walking":
            X_walk.append(X); y_walk.append(y)

    Xw = np.concatenate(X_walk) if X_walk else empty
    yw = np.concatenate(y_walk) if y_walk else np.array([])
    Xi = np.concatenate(X_all)  if X_all  else empty
    yi = np.concatenate(y_all)  if y_all  else np.array([])
    return (Xw, yw), (Xi, yi)

# ═══════════════════════════════════════════════
# TAP
# ═══════════════════════════════════════════════
def process_tap(sess_dir: Path, session_id: str) -> pd.DataFrame:
    """
    Ghép DOWN→UP pairs: tính hold_ms, displacement.
    Chỉ giữ tap có hold_ms ∈ [0, 500) ms.
    """
    rows = []
    for f in sorted(sess_dir.glob("tap_r*.csv")):
        try:
            df = pd.read_csv(f).sort_values("timestamp_ms").reset_index(drop=True)
        except Exception:
            continue
        i = 0
        while i < len(df) - 1:
            if df.at[i, "phase"] != "DOWN":
                i += 1
                continue
            # Tìm UP liền kề (bỏ qua MOVE ở giữa)
            j = i + 1
            while j < len(df) and df.at[j, "phase"] not in ("DOWN", "UP"):
                j += 1
            if j < len(df) and df.at[j, "phase"] == "UP":
                hold = df.at[j, "hold_ms"]
                if 0 <= hold < 500:
                    disp = float(np.sqrt(
                        (df.at[j, "x"] - df.at[i, "x"]) ** 2 +
                        (df.at[j, "y"] - df.at[i, "y"]) ** 2
                    ))
                    rows.append({"session_id": session_id,
                                 "hold_ms":       float(hold),
                                 "displacement":  disp,
                                 "timestamp_ms":  df.at[i, "timestamp_ms"]})
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════
# SCROLL
# ═══════════════════════════════════════════════
def _gesture_features(g: pd.DataFrame) -> dict | None:
    """Features cho 1 scroll gesture (đã sort theo timestamp_ms)."""
    pts = g[["timestamp_ms", "x", "y"]].values.astype(float)
    if len(pts) < 2:
        return None
    t, x, y   = pts[:, 0], pts[:, 1], pts[:, 2]
    dt_total   = t[-1] - t[0]
    if dt_total <= 0:
        return None

    dx      = np.diff(x)
    dy      = np.diff(y)
    dt_seg  = np.clip(np.diff(t), 1e-3, None)     # ms
    seg_d   = np.sqrt(dx ** 2 + dy ** 2)
    v_seg   = seg_d / dt_seg * 1000                # px/s

    traj   = float(seg_d.sum())
    v_mean = traj / dt_total * 1000
    v_max  = float(v_seg.max()) if len(v_seg) else 0.0

    n5       = max(1, len(v_seg) // 20)
    v_start  = float(v_seg[:n5].mean())  if len(v_seg) else 0.0
    v_last5  = float(v_seg[-n5:].mean()) if len(v_seg) else 0.0

    if len(v_seg) >= 2:
        a_seg    = np.diff(v_seg) / dt_seg[1:]
        a_first5 = float(np.abs(a_seg[:n5]).mean()) if len(a_seg) else 0.0
    else:
        a_first5 = 0.0

    angles = np.arctan2(dy, dx)
    mrl    = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))

    return {
        "duration_ms": dt_total,
        "disp_y":      float(y[-1] - y[0]),
        "straight":    float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)),
        "traj":        traj,
        "v_mean":      v_mean,
        "v_max":       v_max,
        "v_start":     v_start,
        "peak_ratio":  v_max / (v_mean + 1e-8),
        "v_last5":     v_last5,
        "mrl":         mrl,
        "a_first5":    a_first5,
        "direction":   float(np.arctan2(y[-1]-y[0], x[-1]-x[0])),
    }

def process_scroll(sess_dir: Path, session_id: str) -> pd.DataFrame:
    """Tái tạo scroll gestures từ raw touch events."""
    gestures = []
    for f in sorted(sess_dir.glob("scroll_r*.csv")):
        try:
            df = pd.read_csv(f).sort_values("timestamp_ms").reset_index(drop=True)
        except Exception:
            continue
        for ptr in df["pointer_id"].unique():
            sub = df[df["pointer_id"] == ptr].reset_index(drop=True)
            in_g, buf = False, []
            for _, row in sub.iterrows():
                if row["phase"] == "DOWN":
                    in_g, buf = True, [row]
                elif in_g:
                    buf.append(row)
                    if row["phase"] == "UP":
                        g   = pd.DataFrame(buf)
                        dur = g["timestamp_ms"].iloc[-1] - g["timestamp_ms"].iloc[0]
                        if len(g) >= 2 and 10 < dur < 5000:
                            feat = _gesture_features(g)
                            if feat:
                                feat["session_id"] = session_id
                                gestures.append(feat)
                        in_g, buf = False, []
    if not gestures:
        return pd.DataFrame()
    cols = ["duration_ms","disp_y","straight","traj","v_mean","v_max",
            "v_start","peak_ratio","v_last5","mrl","a_first5","direction","session_id"]
    return pd.DataFrame(gestures)[cols]

# ═══════════════════════════════════════════════
# KEYSTROKE
# ═══════════════════════════════════════════════
def process_keystroke(sess_dir: Path, session_id: str) -> pd.DataFrame:
    dfs = []
    for f in sorted(sess_dir.glob("keystroke_r*.csv")):
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=["session_id","inter_key_ms","is_delete","timestamp_ms"])
    df = pd.concat(dfs, ignore_index=True)
    df["session_id"] = session_id
    return df[["session_id","inter_key_ms","is_delete","timestamp_ms"]]

# ═══════════════════════════════════════════════
# TOUCH SESSION FEATURES  (48-D)
# ═══════════════════════════════════════════════
def aggregate_touch(tap_df: pd.DataFrame, sc_df: pd.DataFrame,
                    key_df: pd.DataFrame, session_id: str) -> dict:
    feat = {"session_id": session_id}

    # ── TAP (16) ─────────────────────────────────
    tap = tap_df[tap_df["session_id"] == session_id] if not tap_df.empty else pd.DataFrame()
    feat["tap_n"] = len(tap)
    if len(tap) >= 2:
        hold = tap["hold_ms"].values.astype(float)
        disp = tap["displacement"].values.astype(float)
        iti  = np.diff(tap["timestamp_ms"].sort_values().values.astype(float))
    else:
        hold = disp = iti = np.array([])
    feat.update(stats5(hold, "tap_hold"))
    feat.update(stats5(disp, "tap_disp"))
    feat.update(stats5(iti,  "tap_iti"))

    # ── SCROLL (23) ──────────────────────────────
    sc = sc_df[sc_df["session_id"] == session_id] if not sc_df.empty else pd.DataFrame()
    feat["scroll_n"] = len(sc)

    def sc2(col, prefix):
        arr = sc[col].values if len(sc) >= 2 else np.array([])
        m  = float(np.mean(arr))  if len(arr) >= 2 else np.nan
        sd = float(np.std(arr))   if len(arr) >= 2 else np.nan
        return {f"{prefix}_mean": m, f"{prefix}_std": sd}

    for col, pfx in [
        ("duration_ms","scroll_dur"), ("traj","scroll_traj"),
        ("straight","scroll_sdist"), ("v_mean","scroll_vmean"),
        ("v_max","scroll_vmax"),     ("v_last5","scroll_vlast5"),
        ("mrl","scroll_mrl"),        ("a_first5","scroll_afirst5"),
    ]:
        feat.update(sc2(col, pfx))

    if len(sc) >= 2:
        ang = sc["direction"].values
        R   = np.sqrt(np.mean(np.cos(ang))**2 + np.mean(np.sin(ang))**2)
        feat["scroll_dir_circmean"] = float(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))))
        feat["scroll_dir_circstd"]  = float(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1.0))))
    else:
        feat["scroll_dir_circmean"] = feat["scroll_dir_circstd"] = np.nan

    if len(sc) > 0:
        # disp_x tái tạo từ direction + straight (cos θ = disp_x / straight)
        dy = sc["disp_y"].values
        dx = np.cos(sc["direction"].values) * sc["straight"].values
        n  = len(sc)
        feat["scroll_frac_up"]    = float(np.sum((dy < 0) & (np.abs(dy) > np.abs(dx))) / n)
        feat["scroll_frac_down"]  = float(np.sum((dy > 0) & (np.abs(dy) > np.abs(dx))) / n)
        feat["scroll_frac_left"]  = float(np.sum((dx < 0) & (np.abs(dx) > np.abs(dy))) / n)
        feat["scroll_frac_right"] = float(np.sum((dx > 0) & (np.abs(dx) > np.abs(dy))) / n)
    else:
        feat["scroll_frac_up"] = feat["scroll_frac_down"] = \
        feat["scroll_frac_left"] = feat["scroll_frac_right"] = np.nan

    # ── KEYSTROKE (9) ────────────────────────────
    kdf = key_df[key_df["session_id"] == session_id] if not key_df.empty else pd.DataFrame()
    n_total = len(kdf)
    if n_total > 0:
        del_rate = float(kdf["is_delete"].astype(bool).sum() / n_total)
        clean    = kdf[(kdf["inter_key_ms"] > 0) &
                       (~kdf["is_delete"].astype(bool)) &
                       (kdf["inter_key_ms"] < 3000)]
        iki  = clean["inter_key_ms"].values.astype(float)
        n_ch = len(clean)
        if n_ch >= 2:
            dur_s        = (clean["timestamp_ms"].max() - clean["timestamp_ms"].min()) / 1000
            typing_speed = float(n_ch / dur_s) if dur_s > 0 else np.nan
        else:
            typing_speed = np.nan
        burst_rate = float(np.mean(iki < BURST_THR)) if len(iki) else np.nan
    else:
        iki = np.array([])
        del_rate = typing_speed = burst_rate = np.nan
        n_ch = 0

    feat["key_n"] = n_ch
    feat.update(stats5(iki, "key_inter"))
    feat["key_delete_rate"]  = del_rate
    feat["key_typing_speed"] = typing_speed
    feat["key_burst_rate"]   = burst_rate

    return feat

# ── 48 feature columns (cố định thứ tự để match model training) ────────────
TOUCH_COLS = [
    "session_id",
    "tap_n",
    "tap_hold_mean","tap_hold_std","tap_hold_median","tap_hold_p25","tap_hold_p75",
    "tap_disp_mean","tap_disp_std","tap_disp_median","tap_disp_p25","tap_disp_p75",
    "tap_iti_mean","tap_iti_std","tap_iti_median","tap_iti_p25","tap_iti_p75",
    "scroll_n",
    "scroll_dur_mean","scroll_dur_std",
    "scroll_traj_mean","scroll_traj_std",
    "scroll_sdist_mean","scroll_sdist_std",
    "scroll_vmean_mean","scroll_vmean_std",
    "scroll_vmax_mean","scroll_vmax_std",
    "scroll_vlast5_mean","scroll_vlast5_std",
    "scroll_mrl_mean","scroll_mrl_std",
    "scroll_afirst5_mean","scroll_afirst5_std",
    "scroll_dir_circmean","scroll_dir_circstd",
    "scroll_frac_up","scroll_frac_down","scroll_frac_left","scroll_frac_right",
    "key_n",
    "key_inter_mean","key_inter_std","key_inter_median","key_inter_p25","key_inter_p75",
    "key_delete_rate","key_typing_speed","key_burst_rate",
]

# ═══════════════════════════════════════════════
# PER-USER PIPELINE
# ═══════════════════════════════════════════════
def process_user(user_dir: Path):
    uid     = user_dir.name
    out_dir = PROC_DIR / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = sorted(
        [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )
    if not sessions:
        print(f"  [W] {uid}: no sessions")
        return

    print(f"\nProcessing {uid}  ({len(sessions)} sessions) ...")

    all_Xw, all_yw     = [], []   # walking only
    all_Xi, all_yi     = [], []   # all activities
    all_tap, all_sc    = [], []
    all_key            = []
    touch_rows         = []

    for sess_dir in sessions:
        sname = sess_dir.name                    # "session_1"
        label = f"{uid}_{sname}"                 # "user_xxx_session_1"

        (Xw, yw), (Xi, yi) = process_inertial(sess_dir, label)
        tap  = process_tap(sess_dir, sname)
        sc   = process_scroll(sess_dir, sname)
        key  = process_keystroke(sess_dir, sname)

        print(f"  {sname}: walk={len(Xw):>4} sit+stand={len(Xi)-len(Xw):>4} | "
              f"tap={len(tap):>3} scroll={len(sc):>3} key={len(key):>3}")

        if len(Xw): all_Xw.append(Xw); all_yw.append(yw)
        if len(Xi): all_Xi.append(Xi); all_yi.append(yi)
        if len(tap): all_tap.append(tap)
        if len(sc):  all_sc.append(sc)
        if len(key): all_key.append(key)

        touch_rows.append(aggregate_touch(tap, sc, key, sname))

    # ── Save ───────────────────────────────────────
    empty = np.empty((0, WINDOW_SIZE, len(SENSOR_COLS)), np.float64)

    Xw_all = np.concatenate(all_Xw) if all_Xw else empty
    yw_all = np.concatenate(all_yw) if all_yw else np.array([])
    Xi_all = np.concatenate(all_Xi) if all_Xi else empty
    yi_all = np.concatenate(all_yi) if all_yi else np.array([])

    np.save(out_dir / "X_walking.npy",  Xw_all)
    np.save(out_dir / "y_walking.npy",  yw_all)
    np.save(out_dir / "X_inertial.npy", Xi_all)
    np.save(out_dir / "y_inertial.npy", yi_all)

    if all_tap:
        pd.concat(all_tap, ignore_index=True).to_csv(out_dir / "tap_gestures.csv",    index=False)
    if all_sc:
        pd.concat(all_sc,  ignore_index=True).to_csv(out_dir / "scroll_gestures.csv", index=False)

    pd.DataFrame(touch_rows)[TOUCH_COLS].to_csv(
        out_dir / "touch_session_features.csv", index=False)

    print(f"  -> walking={len(Xw_all)}  inertial={len(Xi_all)}  saved to processed/{uid}/")

# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    if not DATA_DIR.is_dir():
        print(f"ERROR: data/ not found at {DATA_DIR}")
        sys.exit(1)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    user_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    if not user_dirs:
        print("No user folders in data/.")
        sys.exit(0)

    print(f"Found {len(user_dirs)} user(s) in {DATA_DIR}")

    for udir in user_dirs:
        process_user(udir)

    print("\n=== Done ===")
