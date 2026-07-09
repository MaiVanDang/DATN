"""
validate_threshold.py — Kiểm chứng ngưỡng PER-OWNER so với ngưỡng cố định.

So sánh trên cùng điểm p_inertial của từng phiên:
  • FIXED   : ngưỡng cố định FIXED_THRESHOLD (0.23) cho mọi owner.
  • PER-OWNER: ngưỡng hiệu chuẩn từ phân bố điểm genuine của chính owner.

Báo cáo FRR (owner bị từ chối — càng thấp càng ổn định) và
FAR (impostor lọt — càng thấp càng an toàn).

Chạy: cd web_demo && python validate_threshold.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from models import load_encoder
from verifier import (
    enroll, verify_session, load_artifacts,
    list_available_users, load_user_inertial, FIXED_THRESHOLD,
)

DATA_DIR  = Path(__file__).parent / "processed"           # testbed nhỏ
ART_DIR   = Path(__file__).parent.parent / "ml_pipeline" / "artifacts" / "cnn" / "export_all"
CKPT      = Path(__file__).parent.parent / "ml_pipeline" / "artifacts" / "cnn" / "models_all" / "backbone.pt"
MODE      = "all"
N_ENROLL  = 4


def main():
    users = list_available_users(DATA_DIR)
    print(f"Users: {users}  |  mode={MODE}  n_enroll={N_ENROLL}\n")
    encoder = load_encoder(str(CKPT), n_users=19, arch="cnn")
    artifacts = load_artifacts(ART_DIR)

    rows = []          # (owner, thr_owner)
    fixed_g, fixed_i = [], []   # quyết định đậu/rớt dưới ngưỡng cố định
    own_g,   own_i   = [], []   # ... dưới ngưỡng per-owner

    for owner in users:
        sess = load_user_inertial(owner, DATA_DIR, mode=MODE)
        keys = sorted(sess.keys())
        if len(keys) < N_ENROLL + 1:
            continue
        en = enroll(owner, N_ENROLL, DATA_DIR, encoder, artifacts, data_mode=MODE)
        thr_o = en.thr_inertial
        rows.append((owner, thr_o))

        # Genuine: các phiên còn lại của owner
        for s in keys[N_ENROLL:]:
            r = verify_session(en, owner, s, DATA_DIR, encoder)
            p = r["p_inertial"]
            if p is None:
                continue
            fixed_g.append(p >= FIXED_THRESHOLD)
            own_g.append(p >= thr_o)

        # Impostor: phiên đầu của mỗi user khác
        for other in users:
            if other == owner:
                continue
            osess = load_user_inertial(other, DATA_DIR, mode=MODE)
            if not osess:
                continue
            s0 = sorted(osess.keys())[0]
            r = verify_session(en, other, s0, DATA_DIR, encoder)
            p = r["p_inertial"]
            if p is None:
                continue
            fixed_i.append(p >= FIXED_THRESHOLD)
            own_i.append(p >= thr_o)

    print("Ngưỡng per-owner đã hiệu chuẩn:")
    for o, t in rows:
        print(f"  {o:>8}: thr = {t:.3f}")

    def rate(decisions, accept_is_correct):
        d = np.asarray(decisions, bool)
        if d.size == 0:
            return float("nan")
        # FRR: genuine bị rớt = 1 - tỉ lệ đậu ; FAR: impostor đậu
        return float((~d).mean()) if accept_is_correct else float(d.mean())

    frr_fixed = rate(fixed_g, True)
    far_fixed = rate(fixed_i, False)
    frr_own   = rate(own_g, True)
    far_own   = rate(own_i, False)

    print("\n=== KẾT QUẢ (mức phiên) ===")
    print(f"{'Ngưỡng':<14}{'FRR(owner rớt)':>16}{'FAR(impostor lọt)':>20}")
    print("-" * 50)
    print(f"{'FIXED 0.23':<14}{frr_fixed*100:>15.1f}%{far_fixed*100:>19.1f}%")
    print(f"{'PER-OWNER':<14}{frr_own*100:>15.1f}%{far_own*100:>19.1f}%")
    print(f"\n  n_genuine={len(own_g)}  n_impostor={len(own_i)}")


if __name__ == "__main__":
    main()
