"""
verifier.py — Lõi xác thực, phiên bản ĐA PHƯƠNG THỨC HIỂN THỊ.

Hiển thị đầy đủ:
  • Điểm quán tính : cos_znorm (mean cosine tới anchor → z-norm cohort → sigmoid)
                     — ĐÂY là đường quyết định khớp bản triển khai on-device.
  • Điểm touch     : Random Forest owner-vs-pool trên đặc trưng 33-D mức
                     sub-window (tap+scroll) — CÙNG sơ đồ với pipeline đánh giá.
  • Điểm fusion    : w·inertial + (1-w)·touch  (w tune tự động trên val, chỉnh được).

verify_session trả về CẢ HAI kết luận:
  • decision_inertial : theo điểm quán tính cos_znorm (bản triển khai)
  • decision_fusion   : theo điểm fusion
=> Hội đồng tự so sánh. Trung thực: nếu fusion không tốt hơn quán tính, bảng
   sẽ cho thấy đúng điều đó (khớp kết luận "fusion không cải thiện open-set").

Chống rò rỉ: cohort/impostor pool loại trừ owner; enroll và test tách phiên.
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from touch_subwindow import touch_subwindows, all_sessions, TOUCH_SUBWIN_DIM as TOUCH_DIM


# ═══════════════════════════════════════════════════════════════════
# Artifacts: pool quán tính (z-norm) + pool touch + scaler touch
# ═══════════════════════════════════════════════════════════════════

class Artifacts:
    def __init__(self, pool_inertial, pool_touch_scaled, touch_mean, touch_scale):
        self.pool_inertial = pool_inertial
        self.pool_touch_scaled = pool_touch_scaled
        self.touch_mean = touch_mean
        self.touch_scale = touch_scale

    def transform_touch(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, np.float64)
        # An toàn: nếu số cột không khớp scaler thì không chuẩn hóa được → báo lệch
        if m.shape[-1] != len(self.touch_mean):
            raise ValueError(
                f"touch dim {m.shape[-1]} != scaler dim {len(self.touch_mean)}")
        return (m - self.touch_mean) / self.touch_scale

    @property
    def has_touch(self) -> bool:
        return len(self.pool_touch_scaled) > 0


def load_artifacts(export_dir: Path) -> Artifacts:
    export_dir = Path(export_dir)
    pool_i = np.load(export_dir / "impostor_pool_inertial.npy")
    assert pool_i.shape[1] == 128, f"Pool inertial dim {pool_i.shape[1]} != 128"

    # Touch là tùy chọn — nếu thiếu file thì chạy thuần quán tính
    pool_t = np.empty((0, TOUCH_DIM), np.float32)
    mean = np.zeros(TOUCH_DIM); scale = np.ones(TOUCH_DIM)
    tp = export_dir / "impostor_pool_touch.npy"
    ts = export_dir / "touch_scaler.json"
    if tp.exists() and ts.exists():
        try:
            _pool_t = np.load(tp).astype(np.float32)
            with open(ts) as f:
                sc = json.load(f)
            _mean = np.asarray(sc["mean"], np.float64)
            _scale = np.asarray(sc["scale"], np.float64)
            # CHỈ bật touch nếu MỌI chiều khớp TOUCH_DIM (33). Lệch → tắt touch an toàn.
            if (_pool_t.ndim == 2 and _pool_t.shape[1] == TOUCH_DIM
                    and len(_mean) == TOUCH_DIM and len(_scale) == TOUCH_DIM):
                pool_t, mean, scale = _pool_t, _mean, _scale
            else:
                print(f"[load_artifacts] CẢNH BÁO: touch scaler/pool lệch chiều "
                      f"(pool={_pool_t.shape}, mean={len(_mean)}, scale={len(_scale)}, "
                      f"cần {TOUCH_DIM}) → TẮT nhánh touch.")
        except Exception as e:
            print(f"[load_artifacts] Lỗi đọc touch artifacts → tắt touch: {e}")
    return Artifacts(pool_i.astype(np.float32), pool_t, mean, scale)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

_FILE_BY_MODE = {
    'walking': ('X_walking.npy', 'y_walking.npy'),
    'all':     ('X_inertial.npy', 'y_inertial.npy'),
}


def list_available_users(data_dir: Path) -> list:
    if not Path(data_dir).exists():
        return []
    out = []
    for d in sorted(Path(data_dir).iterdir()):
        if d.is_dir() and ((d / 'X_inertial.npy').exists() or (d / 'X_walking.npy').exists()):
            out.append(d.name)
    return out


def load_user_inertial(user_id: str, data_dir: Path, mode: str = 'walking') -> dict:
    user_dir = Path(data_dir) / user_id
    x_name, y_name = _FILE_BY_MODE.get(mode, _FILE_BY_MODE['walking'])
    x_path, y_path = user_dir / x_name, user_dir / y_name
    if not x_path.exists():
        alt_x, alt_y = _FILE_BY_MODE['all' if mode == 'walking' else 'walking']
        x_path, y_path = user_dir / alt_x, user_dir / alt_y
        if not x_path.exists():
            return {}
    X = np.load(x_path)
    sess = np.load(y_path, allow_pickle=True)
    out = {}
    prefix = f"{user_id}_"
    for s in np.unique(sess):
        s_str = str(s)
        key = s_str[len(prefix):] if s_str.startswith(prefix) else s_str
        out[key] = X[sess == s].astype(np.float32)
    return out


def _layout(w: np.ndarray) -> np.ndarray:
    if w.ndim != 3:
        raise ValueError(f"Expected 3D, got {w.shape}")
    if w.shape[2] == 9:
        w = w.transpose(0, 2, 1)
    elif w.shape[1] != 9:
        raise ValueError(f"Unexpected shape {w.shape}")
    return w.astype(np.float32)


def _zscore(w: np.ndarray, eps=1e-8) -> np.ndarray:
    return (w - w.mean(2, keepdims=True)) / (w.std(2, keepdims=True) + eps)


def extract_embeddings(encoder: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    windows = _zscore(_layout(windows))
    with torch.no_grad():
        return encoder(torch.from_numpy(windows)).numpy()


# ═══════════════════════════════════════════════════════════════════
# Quán tính: cos_znorm (đường quyết định chính)
# ═══════════════════════════════════════════════════════════════════

SCORE_SCALE = 3.0
SCORE_BIAS  = 2.0

# ─────────────────────────────────────────────────────────────────
# NGƯỠNG CỐ ĐỊNH — calib MỘT LẦN tại điểm EER trên tập đánh giá.
# Giá trị 0.23 lấy từ điểm cân bằng FAR≈FRR trên dữ liệu đánh giá
# (chủ máy đậu hết, ~5% người lạ lọt). Cố định cho MỌI người dùng,
# KHÔNG phụ thuộc số phiên enroll — giống cách FaceID/vân tay calib sẵn.
# Muốn chặt hơn (ưu tiên chặn người lạ) → tăng; muốn dễ nhận chủ máy → giảm.
# ─────────────────────────────────────────────────────────────────
FIXED_THRESHOLD = 0.23

# ─────────────────────────────────────────────────────────────────
# NGƯỠNG THEO TỪNG OWNER (per-owner calibration).
# Thay vì một hằng số chung, ngưỡng được hiệu chuẩn từ PHÂN BỐ điểm
# genuine của chính owner trên phiên val (tách rời anchor → không rò rỉ):
#       thr = mean_genuine − K_STD · std_genuine,  kẹp trong [FLOOR, CEIL].
# Lý do: mỗi owner có độ biến thiên hành vi khác nhau; owner "linh hoạt"
# cần ngưỡng thấp hơn để không bị từ chối nhầm, owner "ổn định" có thể
# đặt ngưỡng cao hơn. Người lạ sau z-norm cohort có điểm ≈0 nên việc hạ
# ngưỡng cho owner gần như không ảnh hưởng khả năng chặn impostor.
# FALLBACK về FIXED_THRESHOLD khi không có phiên val để ước lượng.
# ─────────────────────────────────────────────────────────────────
THR_K_STD      = 1.5   # số độ lệch chuẩn đặt ngưỡng dưới trung bình genuine (khi thiếu impostor)
THR_FLOOR      = 0.02  # sàn an toàn — không cho ngưỡng tụt quá thấp
THR_CEIL       = 0.80  # trần — không cho ngưỡng cao quá gây từ chối owner
THR_AGG_WINDOW = 5     # gộp ~5 cửa sổ trước khi hiệu chuẩn (khớp EWMA triển khai)


def _aggregate_scores(scores, w, seed=0):
    """Gộp điểm theo nhóm w cửa sổ rồi lấy trung bình.

    Lý do: ở mức CỬA SỔ, điểm genuine và impostor chồng lấn nặng; khả năng
    phân biệt chỉ xuất hiện sau khi gộp (giống EWMA lúc vận hành). Vì vậy
    ngưỡng phải được hiệu chuẩn ở ĐÚNG mức gộp mà quyết định sử dụng.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size < w:
        return np.array([s.mean()]) if s.size else s
    s = s.copy()
    np.random.default_rng(seed).shuffle(s)   # cohort pool không theo thời gian
    n = s.size // w
    return s[:n * w].reshape(n, w).mean(axis=1)


def calibrate_owner_threshold(genuine_scores, impostor_scores=None,
                              fallback=FIXED_THRESHOLD):
    """Ngưỡng per-owner cân bằng FAR≈FRR ở mức điểm đã gộp.

    genuine_scores  : điểm s_inertial ∈[0,1] trên cửa sổ genuine của owner
                      (phiên val, tách rời anchor).
    impostor_scores : điểm s_inertial trên cohort impostor (tái dùng embedding).
                      Cả hai được gộp theo THR_AGG_WINDOW trước khi tính ngưỡng,
                      để khớp mức gộp lúc quyết định; nếu thiếu impostor thì lùi
                      K_STD độ lệch chuẩn dưới trung bình genuine.
    """
    g = _aggregate_scores(genuine_scores, THR_AGG_WINDOW, seed=1)
    if g.size < 2:
        return float(fallback)
    if impostor_scores is not None and np.size(impostor_scores) >= THR_AGG_WINDOW:
        imp = _aggregate_scores(impostor_scores, THR_AGG_WINDOW, seed=2)
        y = np.concatenate([np.ones(g.size), np.zeros(imp.size)])
        s = np.concatenate([g, imp])
        thr = _eer_threshold(y, s)          # cân bằng FAR≈FRR cho owner này
    else:
        thr = g.mean() - THR_K_STD * g.std()
    return float(np.clip(thr, THR_FLOOR, THR_CEIL))


def _l2(M): return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)
def mean_cosine(e, a): return (_l2(e) @ _l2(a).T).mean(1)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def fit_cohort(anchors, pool):
    if len(anchors) == 0 or len(pool) == 0:
        return 0.0, 1.0
    s = mean_cosine(pool, anchors)
    return float(s.mean()), float(s.std() + 1e-9)


def score_inertial(embeds, anchors, cohort_mean, cohort_std):
    raw = mean_cosine(embeds, anchors)
    z = (raw - cohort_mean) / cohort_std if cohort_std > 1e-6 else raw
    return sigmoid(SCORE_SCALE * (z - SCORE_BIAS))


# ═══════════════════════════════════════════════════════════════════
# Cohort / impostor pools (loại owner — chống rò rỉ)
# ═══════════════════════════════════════════════════════════════════

def _build_inertial_pool(owner_id, impostor_dir, encoder, artifacts, mode, seed=42):
    embeds = []
    for ud in sorted(Path(impostor_dir).iterdir()):
        if not ud.is_dir() or ud.name == owner_id:
            continue
        try:
            for w in load_user_inertial(ud.name, impostor_dir, mode=mode).values():
                if len(w):
                    embeds.append(extract_embeddings(encoder, w))
        except Exception:
            continue
    if not embeds:
        return artifacts.pool_inertial
    arr = np.concatenate(embeds, 0)
    cap = max(len(artifacts.pool_inertial), 100)
    if len(arr) > cap:
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(len(arr), cap, replace=False)]
    return arr.astype(np.float32)


def _build_touch_pool(owner_id, data_dir, artifacts):
    """Pool người lạ = sub-window touch (33-D) của các user khác owner."""
    vectors = []
    for ud in sorted(Path(data_dir).iterdir()):
        if not ud.is_dir() or ud.name == owner_id:
            continue
        M = touch_subwindows(ud, all_sessions(ud))
        if len(M):
            vectors.append(M)
    if not vectors:
        return artifacts.pool_touch_scaled
    arr = np.vstack(vectors)
    if len(arr) > 100:
        rng = np.random.default_rng(42)
        arr = arr[rng.choice(len(arr), 100, replace=False)]
    return artifacts.transform_touch(arr).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Enrollment
# ═══════════════════════════════════════════════════════════════════

class Enrollment:
    def __init__(self, owner_id, enroll_sessions, anchors, cohort_mean, cohort_std,
                 thr_inertial, rf_touch, artifacts, fusion_w, thr_fusion, data_mode):
        self.owner_id = owner_id
        self.enroll_sessions = enroll_sessions
        self.anchors = anchors
        self.cohort_mean = cohort_mean
        self.cohort_std = cohort_std
        self.thr_inertial = thr_inertial      # ngưỡng cho điểm quán tính
        self.rf_touch = rf_touch               # có thể None
        self.artifacts = artifacts
        self.fusion_w = fusion_w               # trọng số tune tự động (mặc định slider)
        self.thr_fusion = thr_fusion           # ngưỡng cho điểm fusion
        self.data_mode = data_mode


def enroll(owner_id, n_enroll_sessions, data_dir, encoder, artifacts,
           data_mode='walking', seed=42, impostor_dir=None) -> Enrollment:
    if impostor_dir is None:
        impostor_dir = data_dir

    sessions = load_user_inertial(owner_id, data_dir, mode=data_mode)
    keys = sorted(sessions.keys())
    enroll_keys = keys[:n_enroll_sessions]
    if len(enroll_keys) < n_enroll_sessions:
        raise ValueError(f"{owner_id} chỉ có {len(keys)} session; cần {n_enroll_sessions}")

    if len(enroll_keys) >= 2:
        anchor_keys, val_key = enroll_keys[:-1], enroll_keys[-1]
    else:
        anchor_keys, val_key = enroll_keys, None

    # ── Quán tính: anchor + cohort z-norm ──
    anchor_windows = np.concatenate([sessions[s] for s in anchor_keys], 0)
    anchors = extract_embeddings(encoder, anchor_windows)
    cohort = _build_inertial_pool(owner_id, impostor_dir, encoder, artifacts, data_mode, seed)
    cohort_mean, cohort_std = fit_cohort(anchors, cohort)

    # ── Touch: RF owner-vs-pool (sub-window 33-D), nếu có dữ liệu touch ──
    rf_touch = None
    if artifacts.has_touch:
        try:
            ot_raw = touch_subwindows(data_dir / owner_id, set(anchor_keys))
            if len(ot_raw) >= 2:
                ot = artifacts.transform_touch(ot_raw).astype(np.float32)
                pool_t = _build_touch_pool(owner_id, impostor_dir, artifacts)
                if len(pool_t):
                    Xt = np.concatenate([ot, pool_t], 0)
                    yt = np.array([1] * len(ot) + [0] * len(pool_t))
                    rf_touch = RandomForestClassifier(
                        n_estimators=200, max_features='sqrt', class_weight='balanced',
                        min_samples_leaf=1, random_state=seed, n_jobs=-1)
                    rf_touch.fit(Xt, yt)
        except Exception as e:
            print(f"[enroll] Bỏ qua nhánh touch (lỗi/lệch dữ liệu): {e}")
            rf_touch = None

    # ── fusion_w trên val (giữ nguyên — chỉ lấy trọng số, bỏ ngưỡng cũ) ──
    _, fusion_w, _ = _tune(
        owner_id, val_key, data_dir, impostor_dir, encoder, anchors,
        cohort_mean, cohort_std, rf_touch, artifacts, data_mode)

    # ── Ngưỡng PER-OWNER: cân bằng FAR≈FRR riêng cho owner ──
    # Genuine: phiên val (tách rời anchor), chấm ở MỨC CỬA SỔ — sát kịch bản
    # chấm điểm liên tục trên thiết bị. Impostor: cohort pool đã tính sẵn ở
    # trên (tái dùng embedding, không extract thêm).
    thr_inertial = FIXED_THRESHOLD
    if val_key is not None and val_key in sessions and len(sessions[val_key]):
        g_scores = score_inertial(
            extract_embeddings(encoder, sessions[val_key]),
            anchors, cohort_mean, cohort_std)
        imp_scores = (score_inertial(cohort, anchors, cohort_mean, cohort_std)
                      if len(cohort) else None)
        thr_inertial = calibrate_owner_threshold(g_scores, imp_scores)
    thr_fusion = thr_inertial   # dùng chung ngưỡng per-owner cho nhánh fusion

    return Enrollment(owner_id, enroll_keys, anchors, cohort_mean, cohort_std,
                      thr_inertial, rf_touch, artifacts, fusion_w, thr_fusion, data_mode)


def _eer_threshold(y, s):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float(np.clip(thr[i], 0.0, 1.0))


def _touch_score(rf_touch, artifacts, user_dir, sid):
    """Điểm touch mức phiên ∈[0,1] = trung bình prob RF trên các sub-window
    của phiên; None nếu không có/không hợp lệ."""
    if rf_touch is None:
        return None
    M = touch_subwindows(user_dir, {sid})
    if len(M) == 0:
        return None
    try:
        Ms = artifacts.transform_touch(M).astype(np.float32)
        return float(rf_touch.predict_proba(Ms)[:, 1].mean())
    except Exception:
        return None


def _tune(owner_id, val_key, data_dir, impostor_dir, encoder, anchors,
          cohort_mean, cohort_std, rf_touch, artifacts, mode):
    """Trả (thr_inertial, fusion_w, thr_fusion).

    NGƯỠNG dùng hằng số cố định FIXED_THRESHOLD (calib một lần tại EER),
    KHÔNG tune theo val — để độc lập với số phiên enroll, giống triển khai thật.
    Chỉ fusion_w được tune trên val (nếu có touch) vì đó là trọng số, không phải ngưỡng.
    """
    if val_key is None or rf_touch is None:
        return FIXED_THRESHOLD, (1.0 if rf_touch is None else 0.5), FIXED_THRESHOLD

    si, st, y = [], [], []   # điểm quán tính (mức phiên), touch (mức phiên), nhãn

    def add(uid, sid, label, src):
        u = load_user_inertial(uid, src, mode=mode)
        if sid not in u or not len(u[sid]):
            return
        s_in = float(score_inertial(extract_embeddings(encoder, u[sid]),
                                    anchors, cohort_mean, cohort_std).mean())
        s_to = _touch_score(rf_touch, artifacts, Path(src) / uid, sid)
        si.append(s_in); st.append(s_to if s_to is not None else 0.5); y.append(label)

    add(owner_id, val_key, 1, data_dir)
    for u in list_available_users(impostor_dir):
        if u == owner_id:
            continue
        us = load_user_inertial(u, impostor_dir, mode=mode)
        if us:
            add(u, sorted(us.keys())[0], 0, impostor_dir)

    si = np.array(si); st = np.array(st); y = np.array(y)

    if len(np.unique(y)) < 2:
        return FIXED_THRESHOLD, 1.0, FIXED_THRESHOLD

    # grid-search fusion_w (AUC), tie-break về 0.5
    best_w, best_auc, best_d = 0.5, -1.0, 1.0
    for w in np.linspace(0.0, 1.0, 51):
        s = w * si + (1 - w) * st
        try:
            auc = roc_auc_score(y, s)
        except ValueError:
            continue
        d = abs(w - 0.5)
        if auc > best_auc + 1e-6:
            best_auc, best_w, best_d = auc, float(w), d
        elif abs(auc - best_auc) <= 1e-6 and d < best_d:
            best_w, best_d = float(w), d
    return FIXED_THRESHOLD, best_w, FIXED_THRESHOLD


# ═══════════════════════════════════════════════════════════════════
# Verification — trả CẢ 3 điểm + 2 kết luận
# ═══════════════════════════════════════════════════════════════════

def verify_session(enrollment: Enrollment, test_user_id, test_session_id,
                   data_dir, encoder, fusion_w_override=None) -> dict:
    sessions = load_user_inertial(test_user_id, data_dir, mode=enrollment.data_mode)
    is_owner = (test_user_id == enrollment.owner_id)
    base = {'test_user': test_user_id, 'session': test_session_id,
            'is_actual_owner': is_owner}
    if test_session_id not in sessions or len(sessions[test_session_id]) == 0:
        return {**base, 'p_inertial': None, 'p_touch': None, 'p_fusion': None,
                'decision_inertial': 'NO_DATA', 'decision_fusion': 'NO_DATA',
                'correct_inertial': False, 'correct_fusion': False, 'n_windows': 0}

    windows = sessions[test_session_id]
    p_inertial = float(score_inertial(
        extract_embeddings(encoder, windows),
        enrollment.anchors, enrollment.cohort_mean, enrollment.cohort_std).mean())

    p_touch = _touch_score(enrollment.rf_touch, enrollment.artifacts,
                           Path(data_dir) / test_user_id, test_session_id)

    w = enrollment.fusion_w if fusion_w_override is None else fusion_w_override
    p_fusion = (w * p_inertial + (1 - w) * p_touch) if p_touch is not None else p_inertial

    dec_in = 'TRUSTED' if p_inertial >= enrollment.thr_inertial else 'REJECTED'
    dec_fu = 'TRUSTED' if p_fusion   >= enrollment.thr_fusion   else 'REJECTED'

    return {
        **base,
        'p_inertial': p_inertial,
        'p_touch': p_touch,
        'p_fusion': p_fusion,
        'decision_inertial': dec_in,
        'decision_fusion': dec_fu,
        'correct_inertial': (dec_in == 'TRUSTED') == is_owner,
        'correct_fusion':   (dec_fu == 'TRUSTED') == is_owner,
        'n_windows': len(windows),
    }