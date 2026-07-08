"""
touch_subwindow.py — Đặc trưng touch mức SUB-WINDOW (33-D, tap + scroll).

GỘP THỐNG NHẤT: dùng CHÍNH sơ đồ đặc trưng mà pipeline đánh giá
(per_user_eval_cosine) đã dùng để tạo ra kết quả touch ở Chương 5. Nhờ vậy
web_demo trình diễn nhánh touch trên đúng biểu diễn đã được đánh giá, thay vì
một bộ đặc trưng mức-phiên khác.

Đọc tap_gestures.csv + scroll_gestures.csv, cắt cụm theo 20 lần chạm
(kèm 10 lần cuộn) → mỗi cụm là một vector 33 chiều.
"""
import numpy as np
import pandas as pd
from pathlib import Path

TAP_CHUNK = 20
SCR_CHUNK = 10
MIN_TAP_IN_CHUNK = 5
_SCROLL_NUM = ["duration_ms", "traj", "v_mean", "v_max", "straight", "mrl",
               "v_start", "v_last5", "peak_ratio", "a_first5", "disp_y"]
TOUCH_SUBWIN_DIM = 3 * 3 + 1 + len(_SCROLL_NUM) * 2 + 1   # = 33

_raw_cache: dict = {}


def load_raw(user_dir: Path):
    key = str(user_dir)
    if key in _raw_cache:
        return _raw_cache[key]
    try:
        tp = pd.read_csv(Path(user_dir) / "tap_gestures.csv")
        sc = pd.read_csv(Path(user_dir) / "scroll_gestures.csv")
    except Exception:
        _raw_cache[key] = (None, None)
        return None, None
    _raw_cache[key] = (tp, sc)
    return tp, sc


def clear_cache():
    _raw_cache.clear()


def strip_user_prefix(prefixed: str, user_id: str) -> str:
    pre = f"{user_id}_"
    return prefixed[len(pre):] if prefixed.startswith(pre) else prefixed


def all_sessions(user_dir: Path) -> set:
    """Tập session_id (chuỗi) xuất hiện trong tap_gestures.csv của user."""
    tp, _ = load_raw(user_dir)
    if tp is None or "session_id" not in tp.columns:
        return set()
    return set(tp["session_id"].astype(str).unique())


def chunk_feats(tp_c, sc_c) -> np.ndarray:
    f = []
    hold = tp_c["hold_ms"].values if len(tp_c) else np.array([0.0])
    disp = tp_c["displacement"].values if len(tp_c) else np.array([0.0])
    iti = (np.diff(np.sort(tp_c["timestamp_ms"].values))
           if len(tp_c) > 1 else np.array([0.0]))
    for arr in (hold, disp, iti):
        a = arr if len(arr) else np.array([0.0])
        f += [float(np.mean(a)), float(np.std(a)), float(np.median(a))]
    f += [float(len(tp_c))]
    for col in _SCROLL_NUM:
        a = sc_c[col].values if (len(sc_c) and col in sc_c.columns) else np.array([0.0])
        a = a if len(a) else np.array([0.0])
        f += [float(np.mean(a)), float(np.std(a))]
    f += [float(len(sc_c))]
    return np.asarray(f, np.float64)


def touch_subwindows(user_dir: Path, sessions) -> np.ndarray:
    """Trả về (n_subwindow, 33) cho các session chỉ định (session_id chưa tiền tố)."""
    tp, sc = load_raw(user_dir)
    if tp is None:
        return np.zeros((0, TOUCH_SUBWIN_DIM))
    out = []
    for s in {str(x) for x in sessions}:
        t = tp[tp["session_id"].astype(str) == s].reset_index(drop=True)
        k = (sc[sc["session_id"].astype(str) == s].reset_index(drop=True)
             if "session_id" in sc.columns else sc.iloc[0:0])
        if len(t) == 0:
            continue
        n = max(1, len(t) // TAP_CHUNK)
        for i in range(n):
            tc = t.iloc[i * TAP_CHUNK:(i + 1) * TAP_CHUNK]
            kc = k.iloc[i * SCR_CHUNK:(i + 1) * SCR_CHUNK] if len(k) else k
            if len(tc) >= MIN_TAP_IN_CHUNK:
                out.append(chunk_feats(tc, kc))
    return np.asarray(out, np.float64) if out else np.zeros((0, TOUCH_SUBWIN_DIM))
