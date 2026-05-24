"""
touch_train.py — Standalone touch RF training + evaluation.

Quy trình mỗi RUN (giống main.py để metrics so sánh được):
  1. Đọc session list của mỗi user từ y_{walking|inertial}.npy
  2. Session-aware split (train/val/test)
  3. Pre-compute touch vector 48-D 1 lần/run
  4. Fit StandardScaler trên union train vectors
  5. Mỗi owner: sample impostor pool → train RF → score val/test
"""
import argparse, json, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from touch_features import (
    build_session_features, strip_user_prefix, FEAT_DIM, clear_cache,
)
from metrics import (
    compute_auc, compute_eer, compute_far_frr_at_threshold, find_eer_threshold,
)


# ── Defaults ────────────────────────────────────────────────────
DEFAULT_N_RUNS           = 10
DEFAULT_POOL_SIZE        = 100
DEFAULT_TEST_SIZE        = 0.20
DEFAULT_VAL_SIZE         = 0.125
DEFAULT_N_ESTIMATORS     = 200
DEFAULT_MAX_FEATURES     = "sqrt"
DEFAULT_CLASS_WEIGHT     = "balanced"
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_CONTEXT_MODE     = "all"
DEFAULT_SEED_BASE        = 42


def load_session_lists(data_dir: Path, context_mode: str) -> dict:
    y_name = "y_walking.npy" if context_mode == "walking" else "y_inertial.npy"
    out = {}
    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        y_path = user_dir / y_name
        if not y_path.exists():
            print(f"  [skip] {user_dir.name}: thieu {y_name}")
            continue
        sess = np.load(y_path, allow_pickle=True)
        out[user_dir.name] = list(np.unique(sess.astype(str)))
    return out


def split_sessions(sessions_per_user, seed, test_size, val_size):
    rng = np.random.default_rng(seed)
    train_sess, val_sess, test_sess = {}, {}, {}
    for uid, sessions in sessions_per_user.items():
        sessions = np.array(sessions)
        if len(sessions) < 3:
            train_sess[uid] = list(sessions); val_sess[uid] = []; test_sess[uid] = []
            continue
        rng.shuffle(sessions)
        n_te  = max(1, int(len(sessions) * test_size))
        n_val = max(1, int((len(sessions) - n_te) * val_size))
        test_sess[uid]  = list(sessions[:n_te])
        val_sess[uid]   = list(sessions[n_te:n_te + n_val])
        train_sess[uid] = list(sessions[n_te + n_val:])
    return train_sess, val_sess, test_sess


def touch_vectors_for_sessions(data_dir, user_id, session_list):
    user_dir = data_dir / user_id
    vecs = []
    for s in session_list:
        unprefixed = strip_user_prefix(s, user_id)
        vec = build_session_features(user_dir, {unprefixed})
        if vec is not None:
            vecs.append(vec)
    if not vecs:
        return np.zeros((0, FEAT_DIM), dtype=np.float64)
    return np.asarray(vecs, dtype=np.float64)


def precompute(data_dir, sessions_dict):
    return {uid: touch_vectors_for_sessions(data_dir, uid, sess)
            for uid, sess in sessions_dict.items()}


def train_eval_owner(owner_id, train_by_uid, val_by_uid, test_by_uid,
                     scaler, pool_size, n_estimators, max_features,
                     class_weight, min_samples_leaf, seed):
    own_tr = train_by_uid.get(owner_id, np.zeros((0, FEAT_DIM)))
    if len(own_tr) == 0:
        return None, None

    pool_parts = [v for u, v in train_by_uid.items()
                  if u != owner_id and len(v) > 0]
    if not pool_parts:
        return None, None
    pool_all = np.concatenate(pool_parts)
    rng = np.random.default_rng(seed)
    if len(pool_all) > pool_size:
        idx = rng.choice(len(pool_all), size=pool_size, replace=False)
        pool_all = pool_all[idx]

    own_tr_s = scaler.transform(own_tr)
    pool_s   = scaler.transform(pool_all)
    X = np.concatenate([own_tr_s, pool_s])
    y = np.concatenate([np.ones(len(own_tr_s), dtype=np.int32),
                        np.zeros(len(pool_s),  dtype=np.int32)])

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_features=max_features,
        class_weight=class_weight, min_samples_leaf=min_samples_leaf,
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X, y)

    def _xy(by_uid):
        X_list, y_list = [], []
        for uid, vecs in by_uid.items():
            if len(vecs) == 0:
                continue
            X_list.append(vecs)
            y_list.append(np.full(len(vecs), 1 if uid == owner_id else 0, dtype=np.int32))
        if not X_list:
            return None, None
        return np.concatenate(X_list), np.concatenate(y_list)

    val_X,  val_y  = _xy(val_by_uid)
    test_X, test_y = _xy(test_by_uid)
    if val_X is None or test_X is None:
        return None, rf

    val_X_s  = scaler.transform(val_X)
    test_X_s = scaler.transform(test_X)
    s_val    = rf.predict_proba(val_X_s)[:, 1].astype(np.float32)
    s_test   = rf.predict_proba(test_X_s)[:, 1].astype(np.float32)

    auc_val  = compute_auc(val_y,  s_val)
    auc_test = compute_auc(test_y, s_test)
    eer_test = compute_eer(test_y, s_test)
    thr      = find_eer_threshold(test_y, s_test)
    far, frr = compute_far_frr_at_threshold(test_y, s_test, thr)

    return dict(
        owner_id=owner_id, auc_val=float(auc_val), auc_test=float(auc_test),
        eer_test=float(eer_test), far_at_eer=float(far), frr_at_eer=float(frr),
        thr=float(thr), n_train_pos=int(len(own_tr)),
        n_train_neg=int(len(pool_all)), n_val=int(len(val_y)),
        n_test=int(len(test_y)),
    ), rf


def run_once(run_idx, seed, data_dir, sessions_per_user, args, last_run):
    print(f"\n-- RUN {run_idx + 1}/{args.n_runs}  (seed={seed}) --")
    clear_cache()
    train_sess, val_sess, test_sess = split_sessions(
        sessions_per_user, seed, args.test_size, args.val_size
    )
    train_by_uid = precompute(data_dir, train_sess)
    val_by_uid   = precompute(data_dir, val_sess)
    test_by_uid  = precompute(data_dir, test_sess)

    all_train = [v for v in train_by_uid.values() if len(v) > 0]
    if not all_train:
        return [], {}
    scaler = StandardScaler().fit(np.concatenate(all_train))

    rows = []
    saved_rfs = {} if (last_run and args.save_rfs) else None

    for owner_id in sessions_per_user:
        try:
            row, rf = train_eval_owner(
                owner_id, train_by_uid, val_by_uid, test_by_uid,
                scaler, args.pool_size, args.n_estimators,
                args.max_features, args.class_weight,
                args.min_samples_leaf, seed,
            )
        except Exception as e:
            print(f"  [error] {owner_id}: {e}")
            continue

        if row is None:
            continue
        row["run"]  = run_idx
        row["seed"] = seed
        rows.append(row)
        if saved_rfs is not None and rf is not None:
            saved_rfs[owner_id] = (rf, scaler)
        if not args.quiet:
            print(f"  {owner_id:8s}  AUC_v={row['auc_val']:.4f}  "
                  f"AUC_t={row['auc_test']:.4f}  EER={row['eer_test']:.4f}  "
                  f"FAR={row['far_at_eer']:.4f}  FRR={row['frr_at_eer']:.4f}")
    return rows, (saved_rfs or {})


def write_report(df, args, elapsed, output_dir):
    cfg = vars(args).copy()
    cfg.pop("data_dir", None); cfg.pop("output_dir", None)
    cfg.pop("save_rfs", None); cfg.pop("quiet", None)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    per_run = df.groupby("run").agg({
        "auc_test":"mean", "eer_test":"mean",
        "far_at_eer":"mean", "frr_at_eer":"mean",
    })
    lines = [
        "=== TOUCH-ONLY REPORT ===",
        "",
        f"Users      : {sorted(df['owner_id'].unique())}",
        f"Runs       : {args.n_runs}",
        f"Total time : {elapsed/60:.1f} min",
        "",
        "Hyperparameters:",
        f"  context_mode      : {args.context_mode}",
        f"  pool_size         : {args.pool_size}",
        f"  n_estimators      : {args.n_estimators}",
        f"  max_features      : {args.max_features}",
        f"  class_weight      : {args.class_weight}",
        f"  min_samples_leaf  : {args.min_samples_leaf}",
        "",
        "Overall (mean +/- std qua runs):",
        f"  auc_test    : {per_run['auc_test'].mean():.4f} +/- {per_run['auc_test'].std():.4f}",
        f"  eer_test    : {per_run['eer_test'].mean():.4f} +/- {per_run['eer_test'].std():.4f}",
        f"  far_at_eer  : {per_run['far_at_eer'].mean():.4f} +/- {per_run['far_at_eer'].std():.4f}",
        f"  frr_at_eer  : {per_run['frr_at_eer'].mean():.4f} +/- {per_run['frr_at_eer'].std():.4f}",
        "",
        "Per-user (median qua runs, sorted by AUC):",
    ]
    per_user = df.groupby("owner_id").agg(
        auc_med  = ("auc_test", "median"),
        auc_mean = ("auc_test", "mean"),
        auc_std  = ("auc_test", "std"),
        eer_med  = ("eer_test", "median"),
        eer_std  = ("eer_test", "std"),
        n        = ("auc_test", "count"),
    ).sort_values("auc_med")
    lines.append(per_user.to_string())
    report = "\n".join(lines)
    (output_dir / "final_report.txt").write_text(report)
    print("\n" + report)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir",         default="./processed_data")
    p.add_argument("--output-dir",       default="./results_touch")
    p.add_argument("--n-runs",           type=int, default=DEFAULT_N_RUNS)
    p.add_argument("--context-mode",     choices=["walking", "all"], default=DEFAULT_CONTEXT_MODE)
    p.add_argument("--pool-size",        type=int, default=DEFAULT_POOL_SIZE)
    p.add_argument("--n-estimators",     type=int, default=DEFAULT_N_ESTIMATORS)
    p.add_argument("--max-features",     default=DEFAULT_MAX_FEATURES)
    p.add_argument("--class-weight",     default=DEFAULT_CLASS_WEIGHT)
    p.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    p.add_argument("--test-size",        type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--val-size",         type=float, default=DEFAULT_VAL_SIZE)
    p.add_argument("--seed-base",        type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--save-rfs",         action="store_true")
    p.add_argument("--quiet",            action="store_true")
    args = p.parse_args()
    if args.max_features not in ("sqrt", "log2", "auto", None):
        try:    args.max_features = int(args.max_features)
        except ValueError:
            try:    args.max_features = float(args.max_features)
            except ValueError: pass
    if args.class_weight in ("none", "None", ""):
        args.class_weight = None
    return args


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR khong ton tai: {data_dir}")

    print("=" * 70)
    print("TOUCH-ONLY TRAINING")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:18s}: {v}")

    sessions_per_user = load_session_lists(data_dir, args.context_mode)
    if not sessions_per_user:
        raise RuntimeError("Khong load duoc user nao.")
    print(f"\nLoaded {len(sessions_per_user)} users")

    t0 = time.time()
    all_rows = []
    last_rfs = {}
    for run_idx in range(args.n_runs):
        seed = args.seed_base + run_idx
        rows, rfs = run_once(run_idx, seed, data_dir, sessions_per_user, args,
                             last_run=(run_idx == args.n_runs - 1))
        all_rows.extend(rows)
        if rfs:
            last_rfs = rfs

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSaved {output_dir/'summary.csv'} ({len(df)} rows)")

    if args.save_rfs and last_rfs:
        rfs_dir = output_dir / "rfs"
        rfs_dir.mkdir(exist_ok=True)
        for uid, (rf, scaler) in last_rfs.items():
            with (rfs_dir / f"{uid}.pkl").open("wb") as f:
                pickle.dump({"rf": rf, "scaler": scaler}, f)
        print(f"Saved {len(last_rfs)} RFs to {rfs_dir}")

    write_report(df, args, elapsed, output_dir)
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    main()
