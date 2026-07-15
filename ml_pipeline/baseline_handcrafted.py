#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_handcrafted.py — Baseline "truyền thống": đặc trưng thủ công + ML cổ điển
(SVM / Random Forest / kNN), dùng ĐỐI CHỨNG với CNN 1D + cos_znorm của đề tài.

MỤC ĐÍCH
--------
Đại diện cho hướng tiếp cận của các công trình dùng đặc trưng thủ công (ví dụ
IntelliAuth): KHÔNG học biểu diễn, mà trích đặc trưng thống kê/tần số từ cửa sổ
tín hiệu rồi phân loại bằng thuật toán cổ điển. Kết quả dùng làm căn cứ định
lượng cho việc CHỌN MÔ HÌNH (thủ công vs học sâu).

GIAO THỨC (khớp pipeline chính để so sánh công bằng)
----------------------------------------------------
  - Cùng đầu vào : cửa sổ 200x9 trong processed/ -> cả 2 phương pháp thấy CÙNG
                   dữ liệu, chỉ khác cách trích đặc trưng (thủ công vs học).
  - Session-disjoint, chiến lược owner-vs-pool.
  - Tập đóng : impostor = các user KHÁC trong population (trừ HELD_OUT).
  - Tập mở   : impostor = HELD_OUT_IDS (user22..26) CHƯA TỪNG thấy lúc train.
  - Gộp điểm mức phiên bằng EWMA (alpha=0.8, window=5) — giống metrics pipeline.
  - Đo AUC / EER / FAR / FRR; lặp nhiều seed rồi lấy mean +/- std.

LƯU Ý QUAN TRỌNG
----------------
  1. Dữ liệu trong processed/ ĐÃ được z-score theo từng cửa sổ, nên đặc trưng
     biên độ (mean/std/min/max) bị triệt tiêu. Script chỉ dùng đặc trưng BẤT BIẾN
     với z-score: hình dạng phân bố, nhịp (tự tương quan), phổ tần số, tương quan
     chéo kênh. CNN cũng chịu đúng điều kiện này -> so sánh công bằng.
  2. Đây KHÔNG phải bản tái hiện nguyên văn IntelliAuth (bài đó chạy trên bộ dữ
     liệu và đặc trưng biên độ riêng của họ), mà là baseline "đặc trưng thủ công +
     ML cổ điển" chạy trên đúng dữ liệu và giao thức của đề tài.
  3. Kết quả chỉ có ý nghĩa khoa học khi chạy trên DỮ LIỆU THẬT.

CHẠY
----
    python baseline_handcrafted.py --data_dir ./processed --mode all
    python baseline_handcrafted.py --data_dir ./processed --mode walking --clf rf svm
    python baseline_handcrafted.py --data_dir ./processed --mode all --runs 5 --out baseline_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Windows: bảo đảm in được tiếng Việt kể cả khi redirect output ra file/pipe
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── CẤU HÌNH (khớp Active_Auth_Eval_Benchmark.ipynb) ──────────────────────
HELD_OUT_IDS = ["user22", "user23", "user24", "user25", "user26"]
# Tỉ lệ chia phiên — sao ĐÚNG per_user_eval_cosine.py để baseline dùng cùng
# phiên enroll/test với CNN (6 phiên -> 2 enroll / 1 val / 3 test).
ENROLL_FRAC = 0.34
VAL_FRAC = 0.16
TEST_FRAC = 0.50
EWMA_ALPHA = 0.8
EWMA_WINDOW = 5
MAX_WIN_SESS = 60     # cap số cửa sổ lấy mỗi phiên (giữ chi phí tính toán hợp lý)
MAX_POOL_WIN = 1500   # cap tổng số cửa sổ trong pool impostor lúc train
FILE_BY_MODE = {
    "walking": ("X_walking.npy", "y_walking.npy"),
    "all": ("X_inertial.npy", "y_inertial.npy"),
}


# ── TRÍCH ĐẶC TRƯNG THỦ CÔNG (bất biến với z-score) ──────────────────────
def extract_features(X: np.ndarray) -> np.ndarray:
    """Cửa sổ (N, T, C) -> vector đặc trưng thủ công (N, F).

    Chỉ dùng đặc trưng KHÔNG bị z-score triệt tiêu:
      - Hình dạng phân bố : độ lệch (skew), độ nhọn (kurtosis), các phân vị
      - Động học          : tốc độ biến thiên trung bình, tỉ lệ đổi dấu
      - Nhịp              : tự tương quan tại vài độ trễ (bắt chu kỳ bước chân)
      - Phổ tần số        : tần số trội, năng lượng theo dải, entropy phổ
      - Liên kênh         : tương quan Pearson giữa các cặp kênh
    """
    X = np.asarray(X, dtype=np.float64)
    N, T, C = X.shape
    feats = []

    # --- Hình dạng phân bố (per-channel) ---
    for q in (10, 25, 50, 75, 90):
        feats.append(np.percentile(X, q, axis=1))                     # C
    m3 = ((X - X.mean(1, keepdims=True)) ** 3).mean(1)
    m4 = ((X - X.mean(1, keepdims=True)) ** 4).mean(1)
    sd = X.std(1) + 1e-9
    feats.append(m3 / sd ** 3)                                        # skew
    feats.append(m4 / sd ** 4)                                        # kurtosis

    # --- Động học ---
    d = np.diff(X, axis=1)
    feats.append(np.abs(d).mean(1))                                   # tốc độ biến thiên TB
    feats.append(np.abs(d).std(1))
    feats.append((np.diff(np.sign(X), axis=1) != 0).mean(1))          # tỉ lệ đổi dấu

    # --- Nhịp: tự tương quan tại vài độ trễ ---
    Xc = X - X.mean(1, keepdims=True)
    denom = (Xc ** 2).sum(1) + 1e-9
    for lag in (5, 10, 20, 40):
        if lag < T:
            feats.append((Xc[:, :-lag, :] * Xc[:, lag:, :]).sum(1) / denom)

    # --- Phổ tần số ---
    F = np.fft.rfft(Xc, axis=1)
    P = (np.abs(F) ** 2)                                              # (N, T//2+1, C)
    P = P[:, 1:, :]                                                   # bỏ DC
    Ps = P.sum(1) + 1e-9
    feats.append(np.argmax(P, axis=1).astype(np.float64))             # tần số trội
    nb = 4                                                            # năng lượng 4 dải
    edges = np.linspace(0, P.shape[1], nb + 1).astype(int)
    for i in range(nb):
        feats.append(P[:, edges[i]:edges[i + 1], :].sum(1) / Ps)
    Pn = P / Ps[:, None, :]
    feats.append(-(Pn * np.log(Pn + 1e-12)).sum(1))                   # entropy phổ

    # --- Tương quan chéo kênh ---
    Xn = Xc / (np.sqrt((Xc ** 2).mean(1, keepdims=True)) + 1e-9)
    for a in range(C):
        for b in range(a + 1, C):
            feats.append((Xn[:, :, a] * Xn[:, :, b]).mean(1)[:, None])

    out = np.concatenate([f if f.ndim == 2 else f[:, None] for f in feats], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ── METRICS (sao đúng metrics.py của pipeline) ───────────────────────────
def compute_auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return 0.5


def _clean_roc(y, s):
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    m = np.isfinite(thr)
    return fpr[m], tpr[m], thr[m]


def compute_eer(y, s):
    fpr, tpr, thr = _clean_roc(y, s)
    if len(fpr) < 2:
        return 0.5
    fnr = 1.0 - tpr
    diffs = fpr - fnr
    for i in range(len(diffs) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 == 0:
            return float(fpr[i])
        if d0 * d1 < 0:
            t = d0 / (d0 - d1)
            return float(fpr[i] + t * (fpr[i + 1] - fpr[i]))
    idx = int(np.argmin(np.abs(diffs)))
    return float((fpr[idx] + fnr[idx]) / 2)


def find_eer_threshold(y, s):
    fpr, tpr, thr = _clean_roc(y, s)
    if len(thr) < 2:
        return float(np.median(s)) if len(s) else 0.5
    fnr = 1.0 - tpr
    return float(thr[int(np.argmin(np.abs(fpr - fnr)))])


def compute_far_frr_at_threshold(y, s, t):
    p = (s >= t).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return float(fp / (fp + tn + 1e-10)), float(fn / (fn + tp + 1e-10))


def ewma_last(scores, alpha=EWMA_ALPHA, window=EWMA_WINDOW):
    if len(scores) == 0:
        return 0.5
    tail = scores[-window:]
    k = np.arange(len(tail) - 1, -1, -1)
    w = alpha ** k
    return float((w * tail).sum() / w.sum())


def aggregate_per_session(win_scores, session_ids):
    out_s, out_u = [], []
    for s in dict.fromkeys(session_ids.tolist()):
        out_s.append(ewma_last(win_scores[session_ids == s]))
        out_u.append(s)
    return np.asarray(out_s, np.float32), np.asarray(out_u)


# ── DỮ LIỆU ──────────────────────────────────────────────────────────────
def list_users(data_dir: Path):
    return sorted(
        [d.name for d in data_dir.iterdir() if d.is_dir() and (d / "X_inertial.npy").exists()],
        key=lambda u: (len(u), u),
    )


def load_user_features(data_dir: Path, uid: str, mode: str, rng) -> dict:
    """Trả về {session_id: feature_matrix} — đã trích đặc trưng, cap số cửa sổ."""
    xn, yn = FILE_BY_MODE[mode]
    xp, yp = data_dir / uid / xn, data_dir / uid / yn
    if not xp.exists():
        return {}
    X = np.load(xp, mmap_mode="r")
    sess = np.load(yp, allow_pickle=True)
    out = {}
    for s in dict.fromkeys(sess.tolist()):
        idx = np.flatnonzero(sess == s)
        if len(idx) == 0:
            continue
        if len(idx) > MAX_WIN_SESS:
            idx = np.sort(rng.choice(idx, MAX_WIN_SESS, replace=False))
        out[str(s)] = extract_features(np.asarray(X[idx]))
    return out


# ── PHÂN LOẠI ────────────────────────────────────────────────────────────
def make_clf(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=2, random_state=seed, n_jobs=-1)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
            probability=True, random_state=seed))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1))
    raise ValueError(f"clf không hợp lệ: {name}")


def score_windows(clf, F):
    return clf.predict_proba(F)[:, 1].astype(np.float32)


# ── ĐÁNH GIÁ MỘT OWNER ───────────────────────────────────────────────────
def split_sessions(keys, rng):
    """Sao ĐÚNG split_sessions() của per_user_eval_cosine.py.

    Shuffle phiên theo seed rồi chia enroll / val / test theo tỉ lệ, nhờ vậy
    baseline và CNN dùng CÙNG cách chia phiên -> so sánh mới công bằng và
    phương sai giữa các lần chạy mới so được với nhau.
    Trả về (enroll, val, test). Baseline không tune nên val bị bỏ qua.
    """
    ss = np.array(sorted(keys), dtype=object)
    rng.shuffle(ss)
    n = len(ss)
    n_en = max(1, int(round(n * ENROLL_FRAC)))
    en, rest = list(ss[:n_en]), list(ss[n_en:])
    if VAL_FRAC > 0 and rest:
        n_te = max(1, int(round(len(rest) * (TEST_FRAC / (TEST_FRAC + VAL_FRAC + 1e-9)))))
    else:
        n_te = len(rest)
    return en, list(rest[n_te:]), list(rest[:n_te])


def eval_owner(owner, cache, population, impostor_ids, clf_name, seed, rng):
    own = cache.get(owner, {})
    keys = sorted(own.keys())
    if len(keys) < 2:
        return None
    enroll_keys, _val_keys, test_keys = split_sessions(keys, rng)
    if not enroll_keys or not test_keys:
        return None

    # --- train: owner (enroll) vs pool người khác trong population ---
    pos = np.concatenate([own[k] for k in enroll_keys], axis=0)
    neg_parts = [F for uid in population if uid != owner
                 for F in cache.get(uid, {}).values()]
    if not neg_parts:
        return None
    neg = np.concatenate(neg_parts, axis=0)
    if len(neg) > MAX_POOL_WIN:
        neg = neg[rng.choice(len(neg), MAX_POOL_WIN, replace=False)]

    Xtr = np.concatenate([pos, neg], axis=0)
    ytr = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    clf = make_clf(clf_name, seed)
    clf.fit(Xtr, ytr)

    # --- test: phiên còn lại của owner (genuine) + impostor ---
    sc, sid, lab = [], [], []
    for k in test_keys:
        p = score_windows(clf, own[k])
        sc.append(p); sid += [f"{owner}|{k}"] * len(p); lab += [1] * len(p)
    for uid in impostor_ids:
        if uid == owner:
            continue
        for k, F in cache.get(uid, {}).items():
            p = score_windows(clf, F)
            sc.append(p); sid += [f"{uid}|{k}"] * len(p); lab += [0] * len(p)
    if not sc:
        return None

    win_scores = np.concatenate(sc)
    sid = np.asarray(sid); lab = np.asarray(lab)
    sess_scores, sess_uni = aggregate_per_session(win_scores, sid)
    sess_lab = np.asarray([lab[sid == u][0] for u in sess_uni], np.float32)
    if len(np.unique(sess_lab)) < 2:
        return None

    thr = find_eer_threshold(sess_lab, sess_scores)
    far, frr = compute_far_frr_at_threshold(sess_lab, sess_scores, thr)
    return dict(auc=compute_auc(sess_lab, sess_scores),
                eer=compute_eer(sess_lab, sess_scores), far=far, frr=frr)


# ── ĐÁNH GIÁ MỘT KỊCH BẢN ────────────────────────────────────────────────
def run_protocol(cache, population, protocol, clf_name, seed):
    """Sao ĐÚNG giao thức của per_user_eval_cosine.py.

    closed : owner là 21 người trong population; impostor = các owner KHÁC
             trong cùng population.
    open   : owner là 5 danh tính HELD-OUT (backbone/pool chưa từng thấy);
             impostor = các held-out CÒN LẠI — "owner và kẻ tấn công đều là
             người lạ hoàn toàn". Pool âm lúc train vẫn lấy từ population,
             tương ứng cohort/impostor pool đóng gói sẵn trong app.
    """
    rng = np.random.default_rng(seed)
    owners = population if protocol == "closed" else HELD_OUT_IDS
    rows = []
    for owner in owners:
        if protocol == "closed":
            imp = [u for u in population if u != owner]
        else:
            imp = [u for u in HELD_OUT_IDS if u != owner]
        r = eval_owner(owner, cache, population, imp, clf_name, seed, rng)
        if r:
            rows.append(r)
    if not rows:
        return None
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
            for k in ("auc", "eer", "far", "frr")} | {"n_owners": len(rows)}


def run_baseline(data_dir, mode="all", clfs=("rf", "svm", "knn"), runs=3,
                 held_out=None, verbose=True) -> dict:
    """Chạy toàn bộ baseline và trả về dict kết quả.

    Dùng được cả trong notebook (Colab) lẫn CLI.

    data_dir : thư mục processed/
    mode     : 'all' | 'walking'
    clfs     : danh sách bộ phân loại ('rf', 'svm', 'knn')
    runs     : số seed lặp lại
    held_out : danh sách user giữ riêng cho tập mở (None → HELD_OUT_IDS)
    """
    global HELD_OUT_IDS
    if held_out is not None:
        HELD_OUT_IDS = list(held_out)

    data_dir = Path(data_dir)
    users = list_users(data_dir)
    population = [u for u in users if u not in HELD_OUT_IDS]
    if verbose:
        print(f"Users: {len(users)} | population: {len(population)} | "
              f"held-out (tập mở): {HELD_OUT_IDS}")
        print(f"Mode: {mode} | classifiers: {list(clfs)} | runs: {runs}\n")
        print("Trích đặc trưng thủ công...", end=" ", flush=True)

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    cache = {u: load_user_features(data_dir, u, mode, rng) for u in users}
    dim = next((F.shape[1] for d in cache.values() for F in d.values()), 0)
    if verbose:
        print(f"xong ({time.perf_counter()-t0:.0f}s) — {dim} chiều/cửa sổ\n")
        print(f"{'Classifier':<10} {'Kịch bản':<10} {'AUC':>14} {'EER (%)':>14} "
              f"{'FAR (%)':>14} {'FRR (%)':>14}")
        print("-" * 82)

    results = {}
    for clf_name in clfs:
        for protocol in ("closed", "open"):
            rs = [run_protocol(cache, population, protocol, clf_name, s)
                  for s in range(runs)]
            rs = [r for r in rs if r]
            if not rs:
                continue
            agg = {k: (float(np.mean([r[k][0] for r in rs])),
                       float(np.mean([r[k][1] for r in rs])))
                   for k in ("auc", "eer", "far", "frr")}
            results[f"{clf_name}_{protocol}"] = agg
            if verbose:
                lbl = "tập đóng" if protocol == "closed" else "tập mở"
                print(f"{clf_name:<10} {lbl:<10} "
                      f"{agg['auc'][0]:>7.3f}±{agg['auc'][1]:<6.3f} "
                      f"{agg['eer'][0]*100:>7.2f}±{agg['eer'][1]*100:<6.2f} "
                      f"{agg['far'][0]*100:>7.2f}±{agg['far'][1]*100:<6.2f} "
                      f"{agg['frr'][0]*100:>7.2f}±{agg['frr'][1]*100:<6.2f}")

    return {"config": {"mode": mode, "runs": runs, "held_out": HELD_OUT_IDS,
                       "enroll_frac": ENROLL_FRAC, "val_frac": VAL_FRAC,
                       "test_frac": TEST_FRAC, "feature_dim": dim},
            "results": results}


def main():
    ap = argparse.ArgumentParser(description="Baseline đặc trưng thủ công + ML cổ điển")
    ap.add_argument("--data_dir", default="./processed")
    ap.add_argument("--mode", default="all", choices=["all", "walking"])
    ap.add_argument("--clf", nargs="+", default=["rf", "svm", "knn"])
    ap.add_argument("--runs", type=int, default=3, help="số seed lặp lại")
    ap.add_argument("--out", default="baseline_handcrafted_results.json")
    a = ap.parse_args()

    res = run_baseline(a.data_dir, mode=a.mode, clfs=a.clf, runs=a.runs)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nĐã lưu: {a.out}")
    print("\nĐỐI CHIẾU với đề tài (CNN 1D + cos_znorm, cùng giao thức):")
    print("  Tập đóng: EER ~2,24%  |  Tập mở: EER ~10,83%")
    print("  -> Nếu baseline thủ công có EER cao hơn rõ rệt (nhất là tập mở),")
    print("     đó là căn cứ định lượng cho việc chọn CNN học biểu diễn.")


if __name__ == "__main__":
    main()
