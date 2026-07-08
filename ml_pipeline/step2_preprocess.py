import sys
import numpy as np
import pandas as pd
from pathlib import Path

HZ          = 50
WINDOW_SIZE = 4 * HZ
STRIDE      = 20
MAX_GAP_SEC = 5 / HZ
BURST_THR   = 150
ACTIVITIES = ["walking", "sitting", "standing"]

SENSOR_COLS = [
    "acc_x",  "acc_y",  "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x",  "mag_y",  "mag_z",
]

DATA_DIR = Path(__file__).parent / "data"
PROC_DIR = Path(__file__).parent / "processed"

def ts_info(df: pd.DataFrame):
    """Trả về (tên cột timestamp, hệ số quy đổi sang giây)."""
    ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else "timestamp_ns"
    to_sec = 1e3 if ts_col == "timestamp_ms" else 1e9
    return ts_col, to_sec

def split_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    ts_col, to_sec = ts_info(df)
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
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + eps
    return (X - mean) / std

def make_windows(df: pd.DataFrame, label: str):
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

def read_activity(sess_dir: Path, activity: str) -> pd.DataFrame | None:
    dfs = []
    for f in sorted(sess_dir.glob(f"{activity}_att*.csv")):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"    [W] {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

def process_inertial(sess_dir: Path, label: str):
    X_walk, y_walk = [], []
    X_all,  y_all  = [], []
    empty = np.empty((0, WINDOW_SIZE, len(SENSOR_COLS)), np.float64)

    for act in ACTIVITIES:
        df = read_activity(sess_dir, act)
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

def process_tap(sess_dir: Path, session_id: str) -> pd.DataFrame:
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
                    rows.append({
                        "session_id":   session_id,
                        "hold_ms":      float(hold),
                        "displacement": disp,
                        "timestamp_ms": df.at[i, "timestamp_ms"],
                    })
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)

def process_scroll(sess_dir: Path, session_id: str) -> pd.DataFrame:
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
                            feat = gesture_features(g)
                            if feat:
                                feat["session_id"] = session_id
                                gestures.append(feat)
                        in_g, buf = False, []
    if not gestures:
        return pd.DataFrame()
    cols = ["duration_ms", "disp_y", "straight", "traj",
            "v_mean", "v_max", "v_start", "peak_ratio",
            "v_last5", "mrl", "a_first5", "direction", "session_id"]
    return pd.DataFrame(gestures)[cols]

def gesture_features(g: pd.DataFrame) -> dict | None:
    pts = g[["timestamp_ms", "x", "y"]].values.astype(float)
    if len(pts) < 2:
        return None
    t, x, y  = pts[:, 0], pts[:, 1], pts[:, 2]
    dt_total = t[-1] - t[0]
    if dt_total <= 0:
        return None

    dx     = np.diff(x)
    dy     = np.diff(y)
    dt_seg = np.clip(np.diff(t), 1e-3, None)
    seg_d  = np.sqrt(dx ** 2 + dy ** 2)
    v_seg  = seg_d / dt_seg * 1000

    traj   = float(seg_d.sum())
    v_mean = traj / dt_total * 1000
    v_max  = float(v_seg.max()) if len(v_seg) else 0.0

    n5      = max(1, len(v_seg) // 20)
    v_start = float(v_seg[:n5].mean())  if len(v_seg) else 0.0
    v_last5 = float(v_seg[-n5:].mean()) if len(v_seg) else 0.0

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
        "straight":    float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)),
        "traj":        traj,
        "v_mean":      v_mean,
        "v_max":       v_max,
        "v_start":     v_start,
        "peak_ratio":  v_max / (v_mean + 1e-8),
        "v_last5":     v_last5,
        "mrl":         mrl,
        "a_first5":    a_first5,
        "direction":   float(np.arctan2(y[-1] - y[0], x[-1] - x[0])),
    }

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

    all_Xw, all_yw  = [], []
    all_Xi, all_yi  = [], []
    all_tap, all_sc = [], []

    for sess_dir in sessions:
        sname = sess_dir.name
        label = f"{uid}_{sname}"

        (Xw, yw), (Xi, yi) = process_inertial(sess_dir, label)
        tap = process_tap     (sess_dir, sname)
        sc  = process_scroll  (sess_dir, sname)

        print(f"  {sname}: walk={len(Xw):>4} sit+stand={len(Xi)-len(Xw):>4} | "
              f"tap={len(tap):>3} scroll={len(sc):>3}")

        if len(Xw):  all_Xw.append(Xw);  all_yw.append(yw)
        if len(Xi):  all_Xi.append(Xi);  all_yi.append(yi)
        if len(tap): all_tap.append(tap)
        if len(sc):  all_sc.append(sc)

    empty  = np.empty((0, WINDOW_SIZE, len(SENSOR_COLS)), np.float64)
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

    print(f"  -> walking={len(Xw_all)}  inertial={len(Xi_all)}  saved to processed/{uid}/")

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
