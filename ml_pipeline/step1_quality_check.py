import pandas as pd
import numpy as np
from pathlib import Path

# ── CONFIG ─────────────────────────────────────
HZ = 50
MAX_GAP_SEC = 5 / HZ
WINDOW_SIZE = 2 * HZ

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

# ── FORMAT ─────────────────────────────────────
def fmt_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}p{s:02d}s"

def check_status(value, target, unit=""):
    if isinstance(target, float) or unit == "time":
        if value >= target:
            return f"✓ ĐỦ  (dư {fmt_time(value - target)})"
        return f"✗ THIẾU  (cần thêm ~{fmt_time(target - value)})"
    else:
        if value >= target:
            return f"✓ ĐỦ  (dư {value - target})"
        return f"✗ THIẾU  (cần thêm {target - value})"

# ═══════════════════════════════════════════════
# INERTIAL HELPERS
# ═══════════════════════════════════════════════
def split_segments(df: pd.DataFrame, filename: str = "") -> list[pd.DataFrame]:
    """
    Tách DataFrame thành các đoạn liên tục, bỏ gap > MAX_GAP_SEC.
    Hỗ trợ cả timestamp_ms (app mới) và timestamp_ns (app cũ).
    """
    if 'timestamp_ms' in df.columns:
        ts_col, to_sec = 'timestamp_ms', 1e3
    elif 'timestamp_ns' in df.columns:
        ts_col, to_sec = 'timestamp_ns', 1e9
    else:
        return []

    if len(df) < WINDOW_SIZE:
        return []

    df = df.sort_values(ts_col).reset_index(drop=True)
    diffs = df[ts_col].diff() / to_sec

    gap_pos = list(diffs[diffs > MAX_GAP_SEC].index)
    if gap_pos:
        total_gap = diffs[diffs > MAX_GAP_SEC].sum()
        label = f"({filename})" if filename else ""
        print(f"      ⚠  {label} {len(gap_pos)} gap — loại {total_gap:.1f}s")

    boundaries = [0] + gap_pos + [len(df)]
    segments = []
    for i in range(len(boundaries) - 1):
        seg = df.iloc[boundaries[i]:boundaries[i + 1]].copy().reset_index(drop=True)
        seg['_ts_col'] = ts_col
        seg['_to_sec'] = to_sec
        if len(seg) >= WINDOW_SIZE:
            segments.append(seg)
    return segments

def get_duration(df, filename="") -> float:
    """Tính thời gian thu thực tế (giây), loại gap và đoạn quá ngắn."""
    segments = split_segments(df, filename)
    if not segments:
        return 0.0
    total_sec = 0.0
    for seg in segments:
        if len(seg) >= 2:
            ts_col = seg['_ts_col'].iloc[0]
            to_sec = seg['_to_sec'].iloc[0]
            diffs  = seg[ts_col].diff().dropna() / to_sec
            total_sec += diffs[diffs <= MAX_GAP_SEC].sum()
    return total_sec

# ═══════════════════════════════════════════════
# INERTIAL — per session
# ═══════════════════════════════════════════════
def collect_inertial(session_dir: Path) -> dict[str, float]:
    """
    Trả về {activity: total_seconds} cho một session.
    File pattern: walking_att1.csv, standing_att1.csv, ...
    """
    result = {act: 0.0 for act in INERTIAL_TARGET_MIN}
    for act in INERTIAL_TARGET_MIN:
        for f in sorted(session_dir.glob(f"{act}_att*.csv")):
            try:
                df = pd.read_csv(f)
                result[act] += get_duration(df, filename=f.name)
            except Exception as e:
                print(f"      Lỗi {f.name}: {e}")
    return result

# ═══════════════════════════════════════════════
# TOUCH + KEYSTROKE HELPERS — per session
# ═══════════════════════════════════════════════
def count_tap(session_dir: Path) -> int:
    """
    File pattern: tap_r1.csv, tap_r2.csv, ...
    Logic: ghép DOWN→UP liền kề, hold_ms ∈ [0, 500).
    """
    total = 0
    for f in sorted(session_dir.glob("tap_r*.csv")):
        try:
            df = pd.read_csv(f)
            ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
            df = df.sort_values(ts_col).reset_index(drop=True)

            if 'phase' not in df.columns:
                total += len(df)
                continue

            hold_col = 'hold_ms' if 'hold_ms' in df.columns else 'holdMs'
            for i in range(1, len(df)):
                row, prev = df.iloc[i], df.iloc[i - 1]
                if row['phase'] != 'UP' or prev['phase'] != 'DOWN':
                    continue
                hold = row.get(hold_col, -1)
                if 0 <= hold < 500:
                    total += 1
        except Exception as e:
            print(f"      Lỗi tap {f.name}: {e}")
    return total

def count_scroll(session_dir: Path) -> int:
    """
    File pattern: scroll_r1.csv, scroll_r2.csv, ...
    Logic: ghép DOWN→MOVE...→UP theo pointer_id, ≥3 điểm, dt ∈ (10, 5000) ms.
    """
    total = 0
    for f in sorted(session_dir.glob("scroll_r*.csv")):
        try:
            df = pd.read_csv(f)
            ts_col  = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
            ptr_col = 'pointer_id'   if 'pointer_id'   in df.columns else 'pointerId'
            df = df.sort_values(ts_col).reset_index(drop=True)

            if 'phase' not in df.columns or ptr_col not in df.columns:
                continue

            for ptr in df[ptr_col].unique():
                ptr_df = df[df[ptr_col] == ptr]
                in_g, gesture = False, []
                for _, row in ptr_df.iterrows():
                    if row['phase'] == 'DOWN':
                        in_g, gesture = True, [row]
                    elif in_g:
                        gesture.append(row)
                        if row['phase'] == 'UP':
                            if len(gesture) >= 3:
                                g  = pd.DataFrame(gesture)
                                dt = g[ts_col].iloc[-1] - g[ts_col].iloc[0]
                                if 10 < dt < 5000:
                                    total += 1
                            in_g, gesture = False, []
        except Exception as e:
            print(f"      Lỗi scroll {f.name}: {e}")
    return total

def count_keystroke(session_dir: Path) -> int:
    """
    File pattern: keystroke_r1.csv, keystroke_r2.csv, ...
    Logic: bỏ inter_key_ms=0, bỏ is_delete, bỏ outlier >3000 ms.
    """
    dfs = []
    for f in sorted(session_dir.glob("keystroke_r*.csv")):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"      Lỗi keystroke {f.name}: {e}")
    if not dfs:
        return 0

    df = pd.concat(dfs, ignore_index=True)
    key_col = 'inter_key_ms' if 'inter_key_ms' in df.columns else 'interKeyMs'
    del_col = 'is_delete'    if 'is_delete'    in df.columns else 'isDelete'

    df = df[df[key_col] > 0]
    if del_col in df.columns:
        df = df[df[del_col] == False]
    df = df[df[key_col] < 3000]
    return len(df)

def collect_touch(session_dir: Path) -> dict[str, int]:
    """Trả về {metric: count} cho một session."""
    return {
        'tap':       count_tap(session_dir),
        'scroll':    count_scroll(session_dir),
        'keystroke': count_keystroke(session_dir),
    }

# ═══════════════════════════════════════════════
# SESSION CHECK
# ═══════════════════════════════════════════════
def check_session(session_dir: Path) -> tuple[dict, dict]:
    """
    In kết quả kiểm tra cho một session, trả về (inertial_dict, touch_dict)
    để tổng hợp ở cấp user.
    """
    print(f"\n  ┌─ {session_dir.name} ─────────────────")

    inertial = collect_inertial(session_dir)
    touch    = collect_touch(session_dir)

    # ── Inertial ──────────────────────────────
    print("  │  [Inertial]")
    for act, sec in inertial.items():
        print(f"  │    {act:<10}: {fmt_time(sec):>8}")

    # ── Touch / Keystroke ─────────────────────
    print("  │  [Touch / Keystroke]")
    for metric, count in touch.items():
        print(f"  │    {metric:<10}: {count:>5}")

    print("  └" + "─" * 40)
    return inertial, touch

# ═══════════════════════════════════════════════
# USER CHECK  (per-session → tổng kết)
# ═══════════════════════════════════════════════
def check_user(user_dir: Path):
    print("\n" + "=" * 54)
    print(f"  USER: {user_dir.name}")
    print("=" * 54)

    sessions = sorted(
        [d for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("session_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0
    )

    if not sessions:
        print("  ⚠️  Không tìm thấy session nào.")
        return

    # Tích lũy qua từng session
    total_inertial = {act: 0.0 for act in INERTIAL_TARGET_MIN}
    total_touch    = {m: 0    for m  in TOUCH_TARGET}

    for session_dir in sessions:
        inertial, touch = check_session(session_dir)
        for act in total_inertial:
            total_inertial[act] += inertial[act]
        for m in total_touch:
            total_touch[m] += touch[m]

    # ── Tổng kết toàn user ──────────────────────
    print(f"\n  {'━'*52}")
    print(f"  TỔNG KẾT  ({len(sessions)} session)")
    print(f"  {'━'*52}")

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

    verdict = "✓  DỮ LIỆU ĐẦY ĐỦ" if all_ok else "✗  CÒN THIẾU DỮ LIỆU"
    print(f"\n  {verdict}")

# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    data_dir = Path("./data")

    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not user_dirs:
        print("Không tìm thấy user nào trong ./data")
    else:
        for user_dir in user_dirs:
            check_user(user_dir)