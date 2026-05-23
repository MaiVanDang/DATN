"""
verifier.py — Core logic cho web demo (V5-SYNCED FINAL)

Đồng bộ 100% với training pipeline V5:
  • Load impostor pool inertial + touch từ artifacts (export/*.npy)
    → KHÔNG rebuild on-the-fly (đảm bảo pool giống hệt notebook)
  • Load touch_scaler.json từ artifacts → fit on POOL only (giống notebook)
  • Per-window fusion: s_t broadcast per-window, fused per-window rồi mean
  • Tune fusion_w bằng grid search trên held-out enroll session (51 steps,
    tie-break ưu tiên 0.5 — giống fusion.py trong notebook)

Hyperparameters RF giữ y nguyên notebook V5 (config.py):
  n_estimators=200, max_features='sqrt', class_weight='balanced'
  min_samples_leaf=2 (inertial), 1 (touch)
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from touch_features import build_session_features, FEAT_DIM as TOUCH_DIM, FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════
# Artifacts loading (pool + scaler từ notebook V5)
# ═══════════════════════════════════════════════════════════════════

class V5Artifacts:
    """Container cho impostor pool + scaler đã được pre-built từ notebook."""

    def __init__(self,
                 pool_inertial: np.ndarray,
                 pool_touch_scaled: np.ndarray,
                 touch_mean: np.ndarray,
                 touch_scale: np.ndarray):
        self.pool_inertial = pool_inertial
        self.pool_touch_scaled = pool_touch_scaled
        self.touch_mean = touch_mean
        self.touch_scale = touch_scale

    def transform_touch(self, vec_or_matrix: np.ndarray) -> np.ndarray:
        """Apply scaler đã fit từ notebook V5."""
        return (vec_or_matrix - self.touch_mean) / self.touch_scale


def load_artifacts(export_dir: Path) -> V5Artifacts:
    """Load pool inertial + pool touch (đã scaled) + scaler params."""
    export_dir = Path(export_dir)

    pool_i = np.load(export_dir / "impostor_pool_inertial.npy")
    pool_t = np.load(export_dir / "impostor_pool_touch.npy")  # đã scaled

    with open(export_dir / "touch_scaler.json") as f:
        scaler = json.load(f)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)

    assert pool_i.shape[1] == 128, f"Pool inertial dim {pool_i.shape[1]} != 128"
    assert pool_t.shape[1] == TOUCH_DIM, f"Pool touch dim {pool_t.shape[1]} != {TOUCH_DIM}"
    assert len(mean) == TOUCH_DIM, f"Scaler mean dim {len(mean)} != {TOUCH_DIM}"

    return V5Artifacts(
        pool_inertial=pool_i.astype(np.float32),
        pool_touch_scaled=pool_t.astype(np.float32),
        touch_mean=mean,
        touch_scale=scale,
    )


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def list_available_users(data_dir: Path) -> list:
    if not data_dir.exists():
        return []
    users = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and (
            (d / 'X_inertial.npy').exists() or (d / 'X_walking.npy').exists()
        ):
            users.append(d.name)
    return users


def load_user_inertial(user_id: str, data_dir: Path) -> dict:
    """Load inertial data. Dùng X_walking.npy — backbone train trên walking only.
    Fallback về X_inertial.npy nếu không có X_walking.npy."""
    user_dir = data_dir / user_id
    x_path = user_dir / 'X_walking.npy'
    y_path = user_dir / 'y_walking.npy'
    if not x_path.exists():
        x_path = user_dir / 'X_inertial.npy'
        y_path = user_dir / 'y_inertial.npy'
    X = np.load(x_path)
    sess = np.load(y_path, allow_pickle=True)
    sessions = {}
    prefix = f"{user_id}_"
    for s in np.unique(sess):
        s_str = str(s)
        key = s_str[len(prefix):] if s_str.startswith(prefix) else s_str
        mask = sess == s
        sessions[key] = X[mask].astype(np.float32)
    return sessions


def normalize_window_layout(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D, got {windows.shape}")
    n_ch = 9
    if windows.shape[2] == n_ch:          # (N, T, 9) → (N, 9, T)
        windows = windows.transpose(0, 2, 1)
    elif windows.shape[1] == n_ch:        # (N, 9, T) — already channel-first
        pass
    else:
        raise ValueError(f"Unexpected window shape {windows.shape}")
    return windows.astype(np.float32)


def zscore_per_window(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    return (windows - mean) / (std + eps)


# ═══════════════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════════════

def extract_embeddings(encoder: torch.nn.Module, windows: np.ndarray) -> np.ndarray:
    """Input (N, 9, 100) → Output (N, 128)."""
    windows = normalize_window_layout(windows)
    windows = zscore_per_window(windows)
    with torch.no_grad():
        x = torch.from_numpy(windows)
        emb = encoder(x).numpy()
    return emb


# ═══════════════════════════════════════════════════════════════════
# Touch impostor pool — built per-owner, EXCLUDING owner's own data
# ═══════════════════════════════════════════════════════════════════

def _build_touch_impostor_pool(owner_id: str,
                                data_dir: Path,
                                artifacts: V5Artifacts) -> np.ndarray:
    """
    Gom touch vectors từ tất cả users TRỪ owner_id, rồi scale.
    Tránh data leakage: owner data KHÔNG được nằm trong negative pool.
    Fallback về artifacts.pool_touch_scaled nếu không tìm được vector nào.
    """
    vectors = []
    for user_dir in sorted(Path(data_dir).iterdir()):
        if not user_dir.is_dir() or user_dir.name == owner_id:
            continue
        csv_path = user_dir / "touch_session_features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        for c in missing:
            df[c] = 0.0
        mat = df[FEATURE_COLS].to_numpy(dtype=np.float64)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        if len(mat) > 0:
            vectors.append(mat)

    if not vectors:
        return artifacts.pool_touch_scaled  # fallback

    arr = np.vstack(vectors).astype(np.float64)
    # Cap ở 100 vectors — giống notebook POOL_SIZE_TOUCH=100
    if len(arr) > 100:
        rng = np.random.default_rng(42)
        arr = arr[rng.choice(len(arr), size=100, replace=False)]
    return artifacts.transform_touch(arr).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Enrollment
# ═══════════════════════════════════════════════════════════════════

class Enrollment:
    def __init__(self, owner_id, enroll_sessions, rf_inertial, rf_touch,
                 artifacts: V5Artifacts, fusion_w: float,
                 adaptive_threshold: float = 0.5):
        self.owner_id = owner_id
        self.enroll_sessions = enroll_sessions
        self.rf_inertial = rf_inertial
        self.rf_touch = rf_touch
        self.artifacts = artifacts
        self.fusion_w = fusion_w
        self.adaptive_threshold = adaptive_threshold


def _build_inertial_impostor_pool(owner_id: str,
                                   data_dir: Path,
                                   encoder: torch.nn.Module,
                                   artifacts: V5Artifacts,
                                   seed: int = 42) -> np.ndarray:
    """
    Build inertial impostor pool on-the-fly, loại trừ owner_id.
    Tránh data leakage: pool từ notebook V5 chứa embeddings của CẢ owner,
    khiến RF bị confused (owner data xuất hiện ở cả class 0 lẫn class 1).
    Fallback về artifacts.pool_inertial nếu không có user nào khác.
    """
    embeds = []
    for user_dir in sorted(Path(data_dir).iterdir()):
        if not user_dir.is_dir() or user_dir.name == owner_id:
            continue
        if not (user_dir / 'X_inertial.npy').exists() and not (user_dir / 'X_walking.npy').exists():
            continue
        try:
            u_sessions = load_user_inertial(user_dir.name, data_dir)
            for w in u_sessions.values():
                if len(w) > 0:
                    emb = extract_embeddings(encoder, w)
                    embeds.append(emb)
        except Exception:
            continue

    if not embeds:
        return artifacts.pool_inertial  # fallback: chỉ xảy ra nếu chỉ có 1 user

    arr = np.concatenate(embeds, axis=0)
    max_size = len(artifacts.pool_inertial)
    if len(arr) > max_size:
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(len(arr), size=max_size, replace=False)]
    return arr.astype(np.float32)


def enroll(owner_id: str,
           n_enroll_sessions: int,
           data_dir: Path,
           encoder: torch.nn.Module,
           artifacts: V5Artifacts,
           seed: int = 42,
           impostor_dir: Path | None = None) -> Enrollment:
    """Train per-user RF for inertial + touch using V5 pre-built pools.

    impostor_dir : thư mục lấy pool ÂM tính. Mặc định = data_dir (khi owner là
        cohort user). Khi owner là NEWBIE, app phải truyền impostor_dir = cohort
        data_dir riêng, nếu không pool sẽ chỉ chứa newbie khác → RF train sai
        phân phối → cohort user bị nhận nhầm là TRUSTED ở Tab 3.
    """
    if impostor_dir is None:
        impostor_dir = data_dir

    sessions = load_user_inertial(owner_id, data_dir)
    session_keys = sorted(sessions.keys())
    enroll_keys = session_keys[:n_enroll_sessions]
    if len(enroll_keys) < n_enroll_sessions:
        raise ValueError(f"{owner_id} only has {len(session_keys)} sessions; "
                         f"requested {n_enroll_sessions}")

    # Tách session cuối làm val cho fusion_w tuning (không dùng để train RF)
    # → cần ít nhất 4 sessions: RF dùng 3+, val dùng 1
    # → nếu chỉ có ≤3 sessions: dùng tất cả cho RF, fusion_w = 0.5 mặc định
    if len(enroll_keys) >= 4:
        rf_keys = enroll_keys[:-1]
        val_key = enroll_keys[-1]
    else:
        rf_keys  = enroll_keys
        val_key  = None  # không đủ → fusion_w = 0.5

    # ── RF INERTIAL: owner embeddings + on-the-fly pool (loại trừ owner) ──
    # Dùng pool build on-the-fly thay vì artifacts.pool_inertial vì pool từ
    # training chứa embeddings của chính owner → data leakage → P(owner) < 0.5
    owner_windows = np.concatenate([sessions[s] for s in rf_keys], axis=0)
    owner_embeds = extract_embeddings(encoder, owner_windows)

    inertial_pool = _build_inertial_impostor_pool(owner_id, impostor_dir, encoder,
                                                   artifacts, seed)
    X_in = np.concatenate([owner_embeds, inertial_pool], axis=0)
    y_in = np.array([1] * len(owner_embeds) + [0] * len(inertial_pool))

    rf_inertial = RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        class_weight='balanced',
        min_samples_leaf=2,      # V5 config
        random_state=seed,
        n_jobs=-1,
    )
    rf_inertial.fit(X_in, y_in)

    # ── RF TOUCH: owner vecs (scaled bằng V5 scaler) + pre-built scaled pool ──
    user_dir = data_dir / owner_id
    owner_touch_raw = []
    for s in rf_keys:   # chỉ dùng rf_keys, KHÔNG dùng val_key
        v = build_session_features(user_dir, {s})
        if v is not None:
            owner_touch_raw.append(v)

    rf_touch = None
    if len(owner_touch_raw) >= 2:
        owner_touch_arr = np.asarray(owner_touch_raw, dtype=np.float64)
        owner_touch_scaled = artifacts.transform_touch(owner_touch_arr).astype(np.float32)

        # Pool PHẢI loại owner ra — tránh owner data nằm trong negative class
        impostor_touch = _build_touch_impostor_pool(owner_id, impostor_dir, artifacts)
        X_t = np.concatenate([owner_touch_scaled, impostor_touch], axis=0)
        y_t = np.array([1] * len(owner_touch_scaled) + [0] * len(impostor_touch))

        rf_touch = RandomForestClassifier(
            n_estimators=200,
            max_features='sqrt',
            class_weight='balanced',
            min_samples_leaf=1,      # =1 khớp với notebook V5 config
            random_state=seed,
            n_jobs=-1,
        )
        rf_touch.fit(X_t, y_t)

    # ── Tune fusion_w + compute adaptive threshold từ held-out val set ──
    fusion_w, adaptive_threshold = _tune_enrollment_params(
        owner_id=owner_id,
        val_key=val_key,
        data_dir=data_dir,
        encoder=encoder,
        rf_inertial=rf_inertial,
        rf_touch=rf_touch,
        artifacts=artifacts,
        impostor_dir=impostor_dir,
    )

    return Enrollment(owner_id, enroll_keys, rf_inertial, rf_touch,
                      artifacts, fusion_w, adaptive_threshold)


def _find_eer_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Tìm threshold tại điểm EER (Equal Error Rate): FAR ≈ FRR.
    Dùng sklearn roc_curve — trả về threshold ∈ [0, 1].
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    return float(np.clip(thresholds[eer_idx], 0.0, 1.0))


def _tune_enrollment_params(owner_id, val_key, data_dir, encoder,
                             rf_inertial, rf_touch,
                             artifacts: V5Artifacts,
                             impostor_dir: Path | None = None) -> tuple[float, float]:
    """
    Tính (fusion_w, adaptive_threshold) từ held-out val set.

    - fusion_w  : grid-search AUC, 51 steps, tie-break về 0.5
    - adaptive_threshold : threshold tại điểm EER trên val set với fusion_w tối ưu

    Val set = val_key (session held-out, KHÔNG trong RF train) + 1 session/user khác.
    Negatives lấy từ impostor_dir (mặc định = data_dir). Khi owner là newbie,
    impostor_dir phải = cohort data_dir để threshold calibrate đúng phân phối.
    Nếu val_key is None (< 4 enroll sessions) → trả về defaults (0.5, 0.5).
    """
    if val_key is None:
        return (1.0 if rf_touch is None else 0.5), 0.5

    if impostor_dir is None:
        impostor_dir = data_dir

    # ── Build val set ──────────────────────────────────────────────
    val_data = [(owner_id, val_key, 1, data_dir)]
    for u in list_available_users(impostor_dir):
        if u == owner_id:
            continue
        try:
            u_sess = load_user_inertial(u, impostor_dir)
        except Exception:
            continue
        if u_sess:
            val_data.append((u, sorted(u_sess.keys())[0], 0, impostor_dir))

    # ── Collect per-window scores ──────────────────────────────────
    s_i_all, s_t_all, y_all = [], [], []
    for uid, sid, label, src_dir in val_data:
        try:
            u_sess = load_user_inertial(uid, src_dir)
        except Exception:
            continue
        if sid not in u_sess or len(u_sess[sid]) == 0:
            continue

        embeds = extract_embeddings(encoder, u_sess[sid])
        p_i = rf_inertial.predict_proba(embeds)[:, 1]
        n = len(p_i)

        p_t = None
        if rf_touch is not None:
            v_t = build_session_features(src_dir / uid, {sid})
            if v_t is not None:
                v_sc = artifacts.transform_touch(v_t.reshape(1, -1)).astype(np.float32)
                p_t = float(rf_touch.predict_proba(v_sc)[0, 1])

        s_i_all.extend(p_i.tolist())
        s_t_all.extend([p_t if p_t is not None else 0.5] * n)
        y_all.extend([label] * n)

    s_i = np.array(s_i_all, dtype=np.float32)
    s_t = np.array(s_t_all, dtype=np.float32)
    y   = np.array(y_all,   dtype=np.float32)

    if len(np.unique(y)) < 2:
        return (1.0 if rf_touch is None else 0.5), 0.5

    # ── Grid-search fusion_w ───────────────────────────────────────
    if rf_touch is None:
        fusion_w = 1.0
    else:
        best_w, best_auc, best_dist = 0.5, -1.0, 1.0
        for w in np.linspace(0.0, 1.0, 51):
            s = w * s_i + (1 - w) * s_t
            try:
                auc = roc_auc_score(y, s)
            except ValueError:
                continue
            dist = abs(w - 0.5)
            if auc > best_auc + 1e-6:
                best_auc, best_w, best_dist = auc, float(w), dist
            elif abs(auc - best_auc) <= 1e-6 and dist < best_dist:
                best_w, best_dist = float(w), dist
        fusion_w = best_w

    # ── Tính EER threshold từ fused scores với fusion_w tối ưu ────
    fused_val = fusion_w * s_i + (1 - fusion_w) * s_t
    adaptive_threshold = _find_eer_threshold(y, fused_val)

    return fusion_w, adaptive_threshold


# ═══════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════

def verify_session(enrollment: Enrollment,
                   test_user_id: str,
                   test_session_id: str,
                   data_dir: Path,
                   encoder: torch.nn.Module) -> dict:
    """Score 1 session. Fusion per-window rồi aggregate (giống V5)."""

    sessions = load_user_inertial(test_user_id, data_dir)
    if test_session_id not in sessions or len(sessions[test_session_id]) == 0:
        return {
            'test_user': test_user_id, 'session': test_session_id,
            'p_inertial': None, 'p_touch': None, 'fused': None,
            'decision': 'NO_DATA', 'n_windows': 0,
        }

    windows = sessions[test_session_id]
    embeds = extract_embeddings(encoder, windows)
    p_i_per_window = enrollment.rf_inertial.predict_proba(embeds)[:, 1]
    p_inertial = float(p_i_per_window.mean())   # for display

    # Touch score (session-level, broadcast per-window for fusion)
    p_touch = None
    user_dir = data_dir / test_user_id
    v_touch = build_session_features(user_dir, {test_session_id})
    if v_touch is not None and enrollment.rf_touch is not None:
        v_scaled = enrollment.artifacts.transform_touch(
            v_touch.reshape(1, -1)
        ).astype(np.float32)
        p_touch = float(enrollment.rf_touch.predict_proba(v_scaled)[0, 1])

    # Per-window fusion → aggregate
    if p_touch is not None:
        w = enrollment.fusion_w
        fused_per_window = w * p_i_per_window + (1 - w) * p_touch
        fused = float(fused_per_window.mean())
    else:
        fused = p_inertial

    is_owner = (test_user_id == enrollment.owner_id)
    decision = 'TRUSTED' if fused >= enrollment.adaptive_threshold else 'REJECTED'

    return {
        'test_user': test_user_id,
        'session': test_session_id,
        'is_actual_owner': is_owner,
        'p_inertial': p_inertial,
        'p_touch': p_touch,
        'fused': fused,
        'decision': decision,
        'correct': (decision == 'TRUSTED' if is_owner else decision == 'REJECTED'),
        'n_windows': len(windows),
    }


def verify_user_sessions(enrollment: Enrollment, test_user_id: str,
                         session_ids: list, data_dir: Path,
                         encoder: torch.nn.Module) -> pd.DataFrame:
    rows = [verify_session(enrollment, test_user_id, s, data_dir, encoder)
            for s in session_ids]
    return pd.DataFrame(rows)


def verify_batch_impostors(enrollment: Enrollment, data_dir: Path,
                           encoder: torch.nn.Module,
                           n_sessions_per_user: int = 1) -> pd.DataFrame:
    """Test owner's RF against ALL non-owner users."""
    all_rows = []
    other_users = [u for u in list_available_users(data_dir)
                   if u != enrollment.owner_id]
    for u in other_users:
        sessions = load_user_inertial(u, data_dir)
        keys = sorted(sessions.keys())[:n_sessions_per_user]
        for s in keys:
            row = verify_session(enrollment, u, s, data_dir, encoder)
            all_rows.append(row)
    return pd.DataFrame(all_rows)