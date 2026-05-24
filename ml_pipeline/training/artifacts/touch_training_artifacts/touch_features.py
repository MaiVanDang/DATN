"""
touch_features.py — 48-D touch feature schema.

Vector 48 chiều (thứ tự cố định):
  TAP    (16): tap_n, tap_hold x5, tap_disp x5, tap_iti x5
  SCROLL (23): scroll_n, dur x2, traj x2, sdist x2, vmean x2,
               vmax x2, vlast5 x2, mrl x2, afirst5 x2,
               dir_circ x2, frac x4
  KEY     (9): key_n, key_inter x5, delete_rate, typing_speed, burst_rate
"""
import numpy as np
import pandas as pd
from pathlib import Path

TAP_COLS = [
    "tap_n",
    "tap_hold_mean",   "tap_hold_std",   "tap_hold_median", "tap_hold_p25",   "tap_hold_p75",
    "tap_disp_mean",   "tap_disp_std",   "tap_disp_median", "tap_disp_p25",   "tap_disp_p75",
    "tap_iti_mean",    "tap_iti_std",    "tap_iti_median",  "tap_iti_p25",    "tap_iti_p75",
]
SCROLL_COLS = [
    "scroll_n",
    "scroll_dur_mean",    "scroll_dur_std",
    "scroll_traj_mean",   "scroll_traj_std",
    "scroll_sdist_mean",  "scroll_sdist_std",
    "scroll_vmean_mean",  "scroll_vmean_std",
    "scroll_vmax_mean",   "scroll_vmax_std",
    "scroll_vlast5_mean", "scroll_vlast5_std",
    "scroll_mrl_mean",    "scroll_mrl_std",
    "scroll_afirst5_mean","scroll_afirst5_std",
    "scroll_dir_circmean","scroll_dir_circstd",
    "scroll_frac_up", "scroll_frac_down", "scroll_frac_left", "scroll_frac_right",
]
KEY_COLS = [
    "key_n",
    "key_inter_mean", "key_inter_std", "key_inter_median", "key_inter_p25", "key_inter_p75",
    "key_delete_rate", "key_typing_speed", "key_burst_rate",
]

FEATURE_COLS = TAP_COLS + SCROLL_COLS + KEY_COLS
FEAT_DIM     = len(FEATURE_COLS)
assert FEAT_DIM == 48

_csv_cache: dict[Path, pd.DataFrame | None] = {}

def _load_csv(user_dir: Path):
    if user_dir in _csv_cache:
        return _csv_cache[user_dir]
    p = user_dir / "touch_session_features.csv"
    if not p.exists():
        _csv_cache[user_dir] = None
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  Warning: {p}: {e}")
        _csv_cache[user_dir] = None
        return None
    if len(df) == 0 or "session_id" not in df.columns:
        _csv_cache[user_dir] = None
        return None
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    _csv_cache[user_dir] = df
    return df

def clear_cache():
    _csv_cache.clear()

def strip_user_prefix(prefixed: str, user_id: str) -> str:
    pre = f"{user_id}_"
    return prefixed[len(pre):] if prefixed.startswith(pre) else prefixed

def build_session_features(user_dir: Path, session_ids):
    df = _load_csv(user_dir)
    if df is None:
        return None
    targets = {str(s) for s in session_ids}
    matched = df[df["session_id"].astype(str).isin(targets)]
    if len(matched) == 0:
        return None
    mat = matched[FEATURE_COLS].to_numpy(dtype=np.float64)
    vec = mat[0] if len(mat) == 1 else mat.mean(axis=0)
    if np.any(np.isnan(vec)):
        vec = np.nan_to_num(vec, nan=0.0)
    return vec
