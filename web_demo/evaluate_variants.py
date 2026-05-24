"""
evaluate_variants.py — So sánh 6 variants để chọn model nhúng Android.

Chạy: cd web_demo && python evaluate_variants.py

Tối ưu: pre-cache toàn bộ embeddings một lần/variant,
        sau đó tái sử dụng khi evaluate từng owner.
"""
import os, sys, json, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
import torch
from models import load_encoder
from verifier import (
    load_artifacts, load_user_inertial,
    extract_embeddings, list_available_users,
)

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent / "processed_data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
N_ENROLL      = 3      # số session enroll
MAX_IMP_EMB   = 200    # cap impostor pool size
MAX_WIN_SESS  = 50     # max windows lấy từ mỗi session (tránh hàng nghìn windows)
N_JOBS        = -1

VARIANTS = [
    {"name": "CNN + walking",         "arch": "cnn",         "mode": "walking", "dir": "cnn_v2"},
    {"name": "CNN + all",             "arch": "cnn",         "mode": "all",     "dir": "cnn_v2"},
    {"name": "ConvLSTM + walking",    "arch": "convlstm",    "mode": "walking", "dir": "convlstm_v2"},
    {"name": "ConvLSTM + all",        "arch": "convlstm",    "mode": "all",     "dir": "convlstm_v2"},
    {"name": "ConvLSTM-Bi + walking", "arch": "convlstm_bi", "mode": "walking", "dir": "convlstm_bi_v2"},
    {"name": "ConvLSTM-Bi + all",     "arch": "convlstm_bi", "mode": "all",     "dir": "convlstm_bi_v2"},
]

# ── HELPERS ───────────────────────────────────────────────────────────

def find_eer(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


_FILE_BY_MODE = {'walking': ('X_walking.npy', 'y_walking.npy'),
                 'all':     ('X_inertial.npy', 'y_inertial.npy')}

def cache_all_embeddings(encoder, all_users, mode, rng):
    """Load toàn bộ file 1 lần/user (29-88MB), sample ngay, extract embeddings.
    Giữ cấu trúc session để dùng khi evaluate owner."""
    cache = {}
    x_name, y_name = _FILE_BY_MODE.get(mode, _FILE_BY_MODE['walking'])
    for uid in all_users:
        user_dir = DATA_DIR / uid
        x_path = user_dir / x_name
        y_path = user_dir / y_name
        if not x_path.exists():
            alt = 'all' if mode == 'walking' else 'walking'
            x_path = user_dir / _FILE_BY_MODE[alt][0]
            y_path = user_dir / _FILE_BY_MODE[alt][1]
        if not x_path.exists():
            continue

        X    = np.load(x_path).astype(np.float32)   # load 1 lần
        sess = np.load(y_path, allow_pickle=True)

        cache[uid] = {}
        prefix = f"{uid}_"
        for s in np.unique(sess):
            s_str = str(s)
            key   = s_str[len(prefix):] if s_str.startswith(prefix) else s_str
            mask  = (sess == s)
            windows = X[mask]
            if len(windows) == 0:
                continue
            if len(windows) > MAX_WIN_SESS:
                idx = rng.choice(len(windows), MAX_WIN_SESS, replace=False)
                windows = windows[idx]
            cache[uid][key] = extract_embeddings(encoder, windows)
    return cache


def evaluate_variant(variant: dict, all_users: list) -> dict:
    arch    = variant["arch"]
    mode    = variant["mode"]
    art_dir = ARTIFACTS_DIR / variant["dir"] / f"export_{mode}"
    ckpt    = ARTIFACTS_DIR / variant["dir"] / f"models_{mode}" / "backbone.pt"

    meta_path = art_dir / "backbone_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    n_users = len(meta["users_trained"])

    encoder = load_encoder(str(ckpt), n_users=n_users, arch=arch)
    encoder.eval()

    model_size_mb = round(os.path.getsize(ckpt) / 1e6, 2)

    # Latency: 20 forward passes với batch 10 windows
    dummy = torch.zeros(10, 9, 200)
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            encoder(dummy)
    latency_ms = round((time.perf_counter() - t0) / 20 * 1000, 1)

    # Pre-cache toàn bộ embeddings một lần
    print(f"  Caching embeddings...", end=" ", flush=True)
    rng = np.random.default_rng(42)
    emb_cache = cache_all_embeddings(encoder, all_users, mode, rng)
    print(f"done ({len(emb_cache)} users)")

    aucs, eers = [], []

    for owner_id in all_users:
        if owner_id not in emb_cache:
            continue
        owner_cache = emb_cache[owner_id]
        keys = sorted(owner_cache.keys())
        if len(keys) < N_ENROLL + 1:
            continue

        enroll_keys = keys[:N_ENROLL]
        test_keys   = keys[N_ENROLL:]

        # Build impostor pool từ cache (loại owner)
        imp_parts = []
        for uid, u_cache in emb_cache.items():
            if uid == owner_id:
                continue
            for emb in u_cache.values():
                imp_parts.append(emb)
        if not imp_parts:
            continue
        imp_pool = np.concatenate(imp_parts, axis=0)
        if len(imp_pool) > MAX_IMP_EMB:
            idx = rng.choice(len(imp_pool), MAX_IMP_EMB, replace=False)
            imp_pool = imp_pool[idx]

        # Train RF
        owner_emb = np.concatenate([owner_cache[s] for s in enroll_keys], axis=0)
        X_train   = np.concatenate([owner_emb, imp_pool], axis=0)
        y_train   = np.array([1]*len(owner_emb) + [0]*len(imp_pool))

        rf = RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            class_weight="balanced", min_samples_leaf=2,
            random_state=42, n_jobs=N_JOBS,
        )
        rf.fit(X_train, y_train)

        # Collect scores
        scores, labels = [], []

        for s in test_keys:
            p = rf.predict_proba(owner_cache[s])[:, 1].mean()
            scores.append(p); labels.append(1)

        for uid, u_cache in emb_cache.items():
            if uid == owner_id:
                continue
            first_sid = sorted(u_cache.keys())[0]
            p = rf.predict_proba(u_cache[first_sid])[:, 1].mean()
            scores.append(p); labels.append(0)

        if len(np.unique(labels)) < 2:
            continue

        auc = roc_auc_score(labels, scores)
        eer = find_eer(np.array(labels), np.array(scores))
        aucs.append(auc)
        eers.append(eer)

    return {
        "name":          variant["name"],
        "arch":          arch,
        "mode":          mode,
        "auc_mean":      round(float(np.mean(aucs)), 4)  if aucs else None,
        "auc_std":       round(float(np.std(aucs)), 4)   if aucs else None,
        "eer_mean":      round(float(np.mean(eers)), 4)  if eers else None,
        "eer_std":       round(float(np.std(eers)), 4)   if eers else None,
        "model_size_mb": model_size_mb,
        "latency_ms":    latency_ms,
        "n_owners_eval": len(aucs),
    }


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    all_users = list_available_users(DATA_DIR)
    print(f"Users: {len(all_users)}  |  N_enroll={N_ENROLL}\n")
    print(f"{'Variant':<25} {'AUC':>6} {'±':>6} {'EER%':>6} {'±':>6} {'MB':>5} {'ms':>6}")
    print("-" * 65)

    results = []
    for v in VARIANTS:
        print(f"[{v['name']}]")
        r = evaluate_variant(v, all_users)
        results.append(r)

        auc_s = f"{r['auc_mean']:.4f}" if r['auc_mean'] is not None else "   N/A"
        auc_d = f"{r['auc_std']:.4f}"  if r['auc_std']  is not None else "   N/A"
        eer_s = f"{r['eer_mean']*100:.2f}" if r['eer_mean'] is not None else "   N/A"
        eer_d = f"{r['eer_std']*100:.2f}"  if r['eer_std']  is not None else "   N/A"
        print(f"  > {r['name']:<23} {auc_s:>6} {auc_d:>6} {eer_s:>6} {eer_d:>6} "
              f"{r['model_size_mb']:>5} {r['latency_ms']:>6}\n")

    # Lưu JSON
    out = Path(__file__).parent / "evaluation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Tóm tắt
    print("=" * 65)
    valid = [r for r in results if r["eer_mean"] is not None]
    if valid:
        best_eer  = min(valid, key=lambda r: r["eer_mean"])
        best_auc  = max(valid, key=lambda r: r["auc_mean"])
        lightest  = min(valid, key=lambda r: r["model_size_mb"])

        print(f"EER thấp nhất : {best_eer['name']:<25} EER={best_eer['eer_mean']*100:.2f}%  AUC={best_eer['auc_mean']:.4f}")
        print(f"AUC cao nhất  : {best_auc['name']:<25} EER={best_auc['eer_mean']*100:.2f}%  AUC={best_auc['auc_mean']:.4f}")
        print(f"Nhỏ nhất      : {lightest['name']:<25} {lightest['model_size_mb']} MB  latency={lightest['latency_ms']} ms")

    print(f"\nKết quả: {out}")


if __name__ == "__main__":
    main()
