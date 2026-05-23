"""
touch_features.py — Trích vector 48-D từ touch_session_features.csv

ĐÃ ĐỒNG BỘ với step2_preprocess.py (48 features).

Schema 48-D:
    TAP    (16): tap_n, tap_hold×5, tap_disp×5, tap_iti×5
    SCROLL (23): scroll_n, dur×2, traj×2, sdist×2, vmean×2, vmax×2,
                 vlast5×2, mrl×2, afirst5×2, dir_circ×2, frac×4
    KEY     (9): key_n, key_inter×5, key_delete_rate, key_typing_speed,
                 key_burst_rate

Yêu cầu data format:
    processed_data/userX/touch_session_features.csv
        - 1 row per session, 48 feature columns + session_id column
        - File này được tạo bởi step2_preprocess.py

Nếu user folder không có touch_session_features.csv → build_session_features()
trả None. Verifier sẽ tự fallback sang pure-inertial.
"""
import numpy as np
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Schema 48-D (KHỚP 100% với step2_preprocess.py)
# Thứ tự CỐ ĐỊNH — thay đổi ở đây là breaking change cho model.
# ═══════════════════════════════════════════════════════════════════════

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
FEAT_DIM     = len(FEATURE_COLS)   # 48

assert FEAT_DIM == 48, f"FEAT_DIM phải = 48, hiện = {FEAT_DIM}"


# ═══════════════════════════════════════════════════════════════════════
# Cache CSV per user
# ══════════════════════════════════

_session_csv_cache: dict = {}


def _load_session_csv(user_dir: Path):
    """Load touch_session_features.csv, cache theo path."""
    user_dir = Path(user_dir)
    if user_dir in _session_csv_cache:
        return _session_csv_cache[user_dir]

    csv_path = user_dir / "touch_session_features.csv"
    if not csv_path.exists():
        _session_csv_cache[user_dir] = None
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ Lỗi đọc {csv_path}: {e}")
        _session_csv_cache[user_dir] = None
        return None

    if len(df) == 0 or "session_id" not in df.columns:
        _session_csv_cache[user_dir] = None
        return None

    # Bổ sung 0.0 cho cột thiếu (backward-compat)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠ {user_dir.name}: thiếu {len(missing)} cột "
              f"(vd: {missing[:3]}). Điền 0.0.")
        for c in missing:
            df[c] = 0.0

    _session_csv_cache[user_dir] = df
    return df


def clear_cache():
    _session_csv_cache.clear()


def strip_user_prefix(prefixed_session: str, user_id: str) -> str:
    """'user_alice_session_2' → 'session_2'."""
    prefix = f"{user_id}_"
    if prefixed_session.startswith(prefix):
        return prefixed_session[len(prefix):]
    return prefixed_session


# ═══════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════

def build_session_features(user_dir, session_ids):
    """
    Trích vector 48-D trung bình cho tập session_ids của 1 user.
    Trả None nếu user không có touch data hoặc không match session_ids.
    """
    df = _load_session_csv(user_dir)
    if df is None:
        return None

    targets = {str(s) for s in session_ids}
    matched = df[df["session_id"].astype(str).isin(targets)]
    if len(matched) == 0:
        return None

    feat_matrix = matched[FEATURE_COLS].to_numpy(dtype=np.float64)

    if len(feat_matrix) == 1:
        vec = feat_matrix[0]
    else:
        vec = feat_matrix.mean(axis=0)

    if np.any(np.isnan(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec