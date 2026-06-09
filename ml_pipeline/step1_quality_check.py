from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from step2_preprocess import (
    HZ, WINDOW_SIZE,
    split_segments,
    process_tap, process_scroll, process_keystroke,
    aggregate_touch,
)

MAX_GAP_SEC = 5 / HZ

INERTIAL_TARGET_MIN = {
    'walking':  18,
    'standing': 18,
    'sitting':  18,
}

TOUCH_TARGET = {
    'tap':       600,
    'scroll':    600,
    'keystroke': 600,
}

OUTLIER_RATIO_THRESH = 3.0
MIN_SESSIONS_FOR_Z   = 4
MAD_MIN              = 1e-6
TOP_K_COLS           = 5


def fmt_time(sec: float) -> str:
    """Định dạng giây thành 'MpSSs', vd 19.5 → '0p19s'."""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}p{s:02d}s"


def check_status(value, target, unit: str = "") -> str:
    """Trả status string ('✓ ĐỦ' / '✗ THIẾU') kèm chênh lệch."""
    if unit == "time":
        if value >= target:
            return f"✓ ĐỦ  (dư {fmt_time(value - target)})"
        return f"✗ THIẾU  (cần thêm ~{fmt_time(target - value)})"
    else:
        if value >= target:
            return f"✓ ĐỦ  (dư {value - target})"
        return f"✗ THIẾU  (cần thêm {target - value})"


def get_duration(df: pd.DataFrame, filename: str = "") -> float:
    """Thời gian thu thực tế (giây) sau khi loại gap và bỏ segment quá ngắn."""
    segments = split_segments(df)
    if not segments:
        if filename:
            print(f"      ⚠  {filename}: 0 segment hợp lệ (≥ {WINDOW_SIZE} samples)")
        return 0.0

    total_sec = 0.0
    for seg in segments:
        ts_col = 'timestamp_ms' if 'timestamp_ms' in seg.columns else 'timestamp_ns'
        to_sec = 1e3 if ts_col == 'timestamp_ms' else 1e9
        diffs  = seg[ts_col].diff().dropna() / to_sec
        total_sec += diffs[diffs <= MAX_GAP_SEC].sum()
    return total_sec


def collect_inertial(session_dir: Path) -> dict[str, float]:
    """Tổng thời lượng (giây) của 3 activity trong 1 session."""
    result = {act: 0.0 for act in INERTIAL_TARGET_MIN}
    for act in INERTIAL_TARGET_MIN:
        for f in sorted(session_dir.glob(f"{act}_att*.csv")):
            try:
                df = pd.read_csv(f)
                result[act] += get_duration(df, filename=f.name)
            except Exception as e:
                print(f"      Lỗi {f.name}: {e}")
    return result


def collect_touch_counts(
    session_dir: Path, session_id: str
) -> tuple[dict[str, int], pd.DataFrame | None]:
    """Trả (counts theo loại, 48-D feature row của session)."""
    tap_df = process_tap     (session_dir, session_id)
    sc_df  = process_scroll  (session_dir, session_id)
    key_df = process_keystroke(session_dir, session_id)

    counts = {
        'tap':       len(tap_df),
        'scroll':    len(sc_df),
        'keystroke': len(key_df),
    }

    feat_dict = aggregate_touch(tap_df, sc_df, key_df, session_id)
    feat_row  = pd.DataFrame([feat_dict]) if feat_dict else None
    return counts, feat_row


def check_session(session_dir: Path) -> tuple[dict, dict, pd.DataFrame | None]:
    """In kết quả 1 session, trả (inertial_secs, touch_counts, touch_feature_row)."""
    print(f"\n  ┌─ {session_dir.name} ─────────────────")

    inertial = collect_inertial(session_dir)
    counts, feat_row = collect_touch_counts(session_dir, session_dir.name)

    print("  │  [Inertial]")
    for act, sec in inertial.items():
        print(f"  │    {act:<10}: {fmt_time(sec):>8}")

    print("  │  [Touch / Keystroke]")
    for metric, count in counts.items():
        print(f"  │    {metric:<10}: {count:>5}")

    print("  └" + "─" * 40)
    return inertial, counts, feat_row


def detect_outlier_sessions(
    feat_table: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, float]]:
    outliers:   dict[str, dict]  = {}
    all_scores: dict[str, float] = {}
    n = len(feat_table)
    if n < MIN_SESSIONS_FOR_Z:
        return outliers, all_scores

    X   = feat_table.values.astype(np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826

    valid_mask = mad > MAD_MIN
    if not valid_mask.any():
        return outliers, all_scores

    Xs         = (X[:, valid_mask] - med[valid_mask]) / mad[valid_mask]
    valid_cols = feat_table.columns[valid_mask].tolist()
    n_eff_feat = Xs.shape[1]

    for i, sess_id in enumerate(feat_table.index):
        dists = [np.linalg.norm(Xs[i] - Xs[j]) / np.sqrt(n_eff_feat)
                 for j in range(n) if j != i]
        all_scores[sess_id] = float(np.median(dists))

    median_score = max(float(np.median(list(all_scores.values()))), 1e-6)

    for i, sess_id in enumerate(feat_table.index):
        score = all_scores[sess_id]
        ratio = score / median_score
        if ratio < OUTLIER_RATIO_THRESH:
            continue

        abs_z    = np.abs(Xs[i])
        top_idx  = np.argsort(-abs_z)[:TOP_K_COLS]
        top_cols = [(valid_cols[c], float(Xs[i, c])) for c in top_idx]
        outliers[sess_id] = {
            'score':    score,
            'ratio':    ratio,
            'top_cols': top_cols,
        }

    return outliers, all_scores


def check_user(user_dir: Path) -> bool:
    print("\n" + "=" * 54)
    print(f"  USER: {user_dir.name}")
    print("=" * 54)

    sessions = sorted(
        [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )
    if not sessions:
        print("  ⚠️  Không tìm thấy session nào.")
        return False

    total_inertial: dict[str, float] = {act: 0.0 for act in INERTIAL_TARGET_MIN}
    total_touch:    dict[str, int]   = {m: 0    for m   in TOUCH_TARGET}
    feat_rows: list[pd.DataFrame] = []

    for session_dir in sessions:
        inertial, counts, feat_row = check_session(session_dir)
        for act in total_inertial:
            total_inertial[act] += inertial[act]
        for m in total_touch:
            total_touch[m] += counts[m]
        if feat_row is not None and len(feat_row):
            feat_rows.append(feat_row)

    print(f"\n  {'━' * 52}")
    print(f"  TỔNG KẾT  ({len(sessions)} session)")
    print(f"  {'━' * 52}")

    all_ok = True

    print("  [Inertial]")
    for act, sec in total_inertial.items():
        tgt_sec = INERTIAL_TARGET_MIN[act] * 60
        st = check_status(sec, tgt_sec, unit="time")
        if "THIẾU" in st:
            all_ok = False
        print(f"    {act:<10}: {fmt_time(sec):>8} / {fmt_time(tgt_sec)}  {st}")

    print("  [Touch / Keystroke]")
    for metric, count in total_touch.items():
        tgt = TOUCH_TARGET[metric]
        st  = check_status(count, tgt)
        if "THIẾU" in st:
            all_ok = False
        print(f"    {metric:<10}: {count:>5} / {tgt:<5}  {st}")

    print(f"\n  [Outlier check] ({len(feat_rows)} session có touch features)")
    if len(feat_rows) < MIN_SESSIONS_FOR_Z:
        print(f"    ⊘  Chỉ có {len(feat_rows)} session — cần ≥ {MIN_SESSIONS_FOR_Z} "
              f"để median ổn định. Bỏ qua check này.")
    else:
        feat_table = pd.concat(feat_rows, ignore_index=False).set_index('session_id')
        outliers, all_scores = detect_outlier_sessions(feat_table)

        median_score = float(np.median(list(all_scores.values())))
        print(f"    Median session-score = {median_score:.2f}  "
              f"(score càng nhỏ = session càng giống các session khác)")
        for s, sc in sorted(all_scores.items(), key=lambda kv: -kv[1]):
            ratio = sc / max(median_score, 1e-6)
            flag  = "  ← OUTLIER" if s in outliers else ""
            print(f"      {s:<14} score={sc:>5.2f}  (×{ratio:.2f} median){flag}")

        if not outliers:
            print(f"    ✓ Không phát hiện session outlier "
                  f"(ngưỡng: score ≥ {OUTLIER_RATIO_THRESH:.1f}× median).")
        else:
            all_ok = False
            print(f"\n    ✗ {len(outliers)} session lệch hẳn các session khác:")
            for sess_id, info in outliers.items():
                print(f"      • {sess_id}  (score={info['score']:.2f}, "
                      f"= {info['ratio']:.1f}× median, top {len(info['top_cols'])} cột:")
                for col, z in info['top_cols']:
                    sign = "+" if z > 0 else ""
                    print(f"          {col:<28} robust_z = {sign}{z:>6.2f}")
            print(f"\n      → Khuyến nghị: kiểm tra điều kiện thu ở các session trên,")
            print(f"        hoặc thu lại để protocol nhất quán với các session còn lại.")

    verdict = "✓  DỮ LIỆU ĐẦY ĐỦ VÀ NHẤT QUÁN" if all_ok \
              else "✗  CẦN BỔ SUNG / KIỂM TRA LẠI DỮ LIỆU"
    print(f"\n  {verdict}")
    return all_ok


if __name__ == "__main__":
    data_dir  = Path("./data")
    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not user_dirs:
        print(f"Không tìm thấy user nào trong {data_dir}")
    else:
        n_pass = n_fail = 0
        for user_dir in user_dirs:
            if check_user(user_dir):
                n_pass += 1
            else:
                n_fail += 1

        print("\n" + "=" * 54)
        print(f"  TỔNG: {n_pass} user PASS, {n_fail} user cần xem lại")
        print("=" * 54)
