"""
build_pool.py — Rebuild impostor_pool_touch.npy + touch_scaler.json (48-D)

Run từ thư mục demo/:
    python build_pool.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from touch_features import FEATURE_COLS, FEAT_DIM

DATA_DIR   = Path(__file__).parent / "processed_data"
EXPORT_DIR = Path(__file__).parent / "export"
EXPORT_DIR.mkdir(exist_ok=True)

print(f"FEAT_DIM = {FEAT_DIM} (expected 48)")
assert FEAT_DIM == 48

rows = []
for user_dir in sorted(DATA_DIR.iterdir()):
    if not user_dir.is_dir():
        continue
    csv_path = user_dir / "touch_session_features.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {user_dir.name}: no touch_session_features.csv")
        continue
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] {user_dir.name}: missing {missing[:3]}...")
        for c in missing:
            df[c] = 0.0
    mat = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    rows.append(mat)
    print(f"  {user_dir.name}: {len(mat)} sessions")

pool_raw = np.vstack(rows).astype(np.float64)
print(f"\nPool raw shape: {pool_raw.shape}")

scaler = StandardScaler()
pool_scaled = scaler.fit_transform(pool_raw).astype(np.float32)

out_npy = EXPORT_DIR / "impostor_pool_touch.npy"
np.save(out_npy, pool_scaled)
print(f"Saved {out_npy}  shape={pool_scaled.shape}")

scaler_dict = {
    "mean":  scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}
out_json = EXPORT_DIR / "touch_scaler.json"
with open(out_json, "w") as f:
    json.dump(scaler_dict, f)
print(f"Saved {out_json}  mean[0]={scaler.mean_[0]:.4f}")

print("\nDone.")
