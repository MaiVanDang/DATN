import os
import sys
import numpy as np
import pandas as pd

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed")

if not os.path.isdir(PROCESSED_DIR):
    sys.exit(f"[ERROR] Không tìm thấy thư mục: {PROCESSED_DIR}\n"
             f"        Hãy chỉnh PROCESSED_DIR trong script.")

# ── Hằng số tên file ─────────────────────────────────────────────────────────
F_SESS   = "touch_session_features.csv"
F_TAP    = "tap_gestures.csv"
F_SCROLL = "scroll_gestures.csv"
F_XI     = "X_inertial.npy"
F_XW     = "X_walking.npy"
F_YI     = "y_inertial.npy"
F_YW     = "y_walking.npy"

# ── Duyệt qua tất cả user folder ─────────────────────────────────────────────
user_dirs = sorted([
    d for d in os.listdir(PROCESSED_DIR)
    if os.path.isdir(os.path.join(PROCESSED_DIR, d))
])

if not user_dirs:
    sys.exit(f"[ERROR] Không có thư mục user nào trong: {PROCESSED_DIR}")

# ── Collect dữ liệu ──────────────────────────────────────────────────────────
all_sess_rows  = []
imu_summary    = []
imu_window_shape = None

total_inertial = 0
total_walking  = 0

for user_id in user_dirs:
    user_path = os.path.join(PROCESSED_DIR, user_id)

    # ── IMU ──────────────────────────────────────────────────────────────────
    n_iner = n_walk = 0

    path_xi = os.path.join(user_path, F_XI)
    path_yi = os.path.join(user_path, F_YI)
    path_xw = os.path.join(user_path, F_XW)
    path_yw = os.path.join(user_path, F_YW)

    if os.path.exists(path_yi):
        yi = np.load(path_yi, allow_pickle=True)
        n_iner = len(yi)
        total_inertial += n_iner
        if imu_window_shape is None and os.path.exists(path_xi):
            xi = np.load(path_xi, allow_pickle=True)
            if xi.ndim == 3:
                imu_window_shape = (xi.shape[1], xi.shape[2])

    if os.path.exists(path_yw):
        yw = np.load(path_yw, allow_pickle=True)
        n_walk = len(yw)
        total_walking += n_walk

    imu_summary.append({
        "user_id":          user_id,
        "inertial_samples": n_iner,
        "walking_samples":  n_walk,
        "total_imu":        n_iner + n_walk,
    })

    # ── Touch / Keystroke ────────────────────────────────────────────────────
    path_sess = os.path.join(user_path, F_SESS)
    if os.path.exists(path_sess):
        df = pd.read_csv(path_sess)
        df.insert(0, "user_id", user_id)
        all_sess_rows.append(df)
    else:
        print(f"  [WARN] Không tìm thấy {F_SESS} cho {user_id}")

# ── Ghép toàn bộ session ─────────────────────────────────────────────────────
if not all_sess_rows:
    sys.exit("[ERROR] Không đọc được bất kỳ touch_session_features.csv nào.")

sess_df  = pd.concat(all_sess_rows, ignore_index=True)
imu_df   = pd.DataFrame(imu_summary)

# ── In kết quả ───────────────────────────────────────────────────────────────
W = 62

print("=" * W)
print("  DATASET STATISTICS")
print("=" * W)

# ── [1] IMU ───────────────────────────────────────────────────────────────────
print("\n[1] CẢM BIẾN IMU — theo User")
print("-" * W)
print(f"  {'User':<12} {'Inertial':>10} {'Walking':>10} {'Total IMU':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
for _, r in imu_df.iterrows():
    print(f"  {r['user_id']:<12} {r['inertial_samples']:>10,} "
          f"{r['walking_samples']:>10,} {r['total_imu']:>10,}")

total_imu = total_inertial + total_walking
print(f"  {'TOTAL':<12} {total_inertial:>10,} {total_walking:>10,} {total_imu:>10,}")
if imu_window_shape:
    print(f"\n  ► Shape mỗi mẫu: {imu_window_shape[0]} timesteps × "
          f"{imu_window_shape[1]} sensor axes")

# ── [2] Touch & Keystroke theo User ──────────────────────────────────────────
print("\n[2] TOUCH & KEYSTROKE — theo User")
print("-" * W)

touch_by_user = (
    sess_df
    .groupby("user_id")[["tap_n", "scroll_n", "key_n"]]
    .sum()
    .astype(int)
    .rename(columns={"tap_n": "taps", "scroll_n": "scrolls", "key_n": "keystrokes"})
)
touch_by_user["touch(tap+scroll)"] = touch_by_user["taps"] + touch_by_user["scrolls"]
touch_by_user["total"]             = touch_by_user["touch(tap+scroll)"] + touch_by_user["keystrokes"]

print(f"  {'User':<12} {'Taps':>8} {'Scrolls':>9} {'Keystrokes':>12} "
      f"{'Touch':>8} {'Total':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*12} {'-'*8} {'-'*8}")
for uid, r in touch_by_user.iterrows():
    print(f"  {uid:<12} {r['taps']:>8,} {r['scrolls']:>9,} "
          f"{r['keystrokes']:>12,} {r['touch(tap+scroll)']:>8,} {r['total']:>8,}")

gt = touch_by_user.sum()
print(f"  {'TOTAL':<12} {gt['taps']:>8,} {gt['scrolls']:>9,} "
      f"{gt['keystrokes']:>12,} {gt['touch(tap+scroll)']:>8,} {gt['total']:>8,}")

# ── [3] Chi tiết theo Session ─────────────────────────────────────────────────
print("\n[3] CHI TIẾT THEO SESSION")
print("-" * W)

det = sess_df[["user_id", "session_id", "tap_n", "scroll_n", "key_n"]].copy()
det = det.astype({"tap_n": int, "scroll_n": int, "key_n": int})
det["total"] = det["tap_n"] + det["scroll_n"] + det["key_n"]

print(f"  {'User':<12} {'Session':<14} {'Taps':>6} {'Scrolls':>8} "
      f"{'Keys':>6} {'Total':>7}")
print(f"  {'-'*12} {'-'*14} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")
for _, r in det.iterrows():
    print(f"  {r['user_id']:<12} {r['session_id']:<14} {r['tap_n']:>6} "
          f"{r['scroll_n']:>8} {r['key_n']:>6} {r['total']:>7}")

# ── [4] Tóm tắt ──────────────────────────────────────────────────────────────
print("\n" + "=" * W)
print("  TÓM TẮT TỔNG QUAN")
print("=" * W)
print(f"  Tổng số User                : {len(user_dirs)}")
print(f"  Tổng số Session             : {sess_df['session_id'].nunique()}")
print(f"  Tổng mẫu IMU — Inertial     : {total_inertial:,}")
print(f"  Tổng mẫu IMU — Walking      : {total_walking:,}")
print(f"  Tổng mẫu IMU (tất cả)       : {total_imu:,}")
print(f"  Tổng Touch  (tap + scroll)  : {int(gt['touch(tap+scroll)']):,}")
print(f"  Tổng Keystroke              : {int(gt['keystrokes']):,}")
print(f"  Tổng Touch + Keystroke      : {int(gt['total']):,}")
print("=" * W)