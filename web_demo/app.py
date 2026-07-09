"""
app.py — Active Auth Demo (đa phương thức hiển thị)

Hiển thị đủ 3 điểm + 2 kết luận để hội đồng tự so sánh:
  Điểm quán tính (cos_znorm) · Điểm touch (RF) · Điểm fusion
  Kết luận theo quán tính (= bản triển khai) · Kết luận theo fusion

Đường quyết định CHÍNH của hệ thống là quán tính cos_znorm (đơn modal on-device);
fusion hiển thị để minh bạch — nếu fusion không tốt hơn, bảng sẽ cho thấy đúng vậy.

Chạy:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from models import load_encoder
from verifier import (
    list_available_users, load_user_inertial,
    enroll, verify_session, load_artifacts,
)

ARTIFACTS_ROOT = Path("artifacts")
ARCH, ARCH_DIR, N_USERS = "cnn", "cnn", 19
MODE_LABEL = {"walking": "Đi bộ", "all": "Toàn bộ hoạt động"}

st.set_page_config(page_title="Active Auth Demo", page_icon="🔐", layout="wide")
st.title("🔐 Demo xác thực hành vi — đa phương thức")
st.caption("Đường quyết định on-device là **quán tính (cos_znorm)**. Điểm touch và "
           "fusion hiển thị để so sánh minh bạch.")


@st.cache_resource
def get_encoder(ckpt): return load_encoder(ckpt, n_users=N_USERS, arch=ARCH)
@st.cache_resource
def get_artifacts(d): return load_artifacts(Path(d))


def fmt(x): return "—" if x is None else f"{x:.3f}"


# ── Sidebar: cấu hình + enroll + slider fusion ──
with st.sidebar:
    st.header("⚙️ Cấu hình")
    data_dir = Path(st.text_input("Thư mục dữ liệu", value="processed_data"))
    mode = st.radio("Ngữ cảnh", ["all", "walking"],
                    format_func=lambda m: MODE_LABEL[m], index=0)

    export_dir = ARTIFACTS_ROOT / ARCH_DIR / f"export_{mode}"
    ckpt = ARTIFACTS_ROOT / ARCH_DIR / f"models_{mode}" / "backbone.pt"

    users = list_available_users(data_dir)
    if not users:
        st.error(f"Không thấy user trong `{data_dir}`."); st.stop()
    st.success(f"✓ {len(users)} user")

    encoder = get_encoder(str(ckpt))
    artifacts = get_artifacts(str(export_dir))

    st.divider()
    st.header("👤 Đăng ký chủ máy")
    owner_id = st.selectbox("Chủ máy", users)
    n_sess = len(load_user_inertial(owner_id, data_dir, mode=mode))
    n_enroll = st.slider("Số session enroll", 1, max(1, n_sess), min(2, n_sess))

    if st.button("🎯 Đăng ký", type="primary", use_container_width=True):
        with st.spinner("Đang tạo hồ sơ..."):
            try:
                enr = enroll(owner_id, n_enroll, data_dir, encoder, artifacts, data_mode=mode)
                st.session_state["enr"] = enr
                st.session_state["users"] = users
                st.session_state["dir"] = str(data_dir)
                has_touch = enr.rf_touch is not None
                st.success(f"✓ `{owner_id}` — {len(enr.anchors)} anchor"
                           + (f", touch ✓, fusion_w(auto)={enr.fusion_w:.2f}" if has_touch
                              else ", touch ✗ (thiếu dữ liệu chạm)"))
            except Exception as e:
                st.error(f"Đăng ký thất bại: {e}")

enr = st.session_state.get("enr")
if enr is not None and enr.rf_touch is not None:
    with st.sidebar:
        st.divider()
        st.header("🎚️ Fusion")
        st.caption(f"Mặc định = tune tự động trên val ({enr.fusion_w:.2f}). "
                   "Slider chỉ để khám phá — đừng chỉnh để 'làm đẹp số'.")
        fusion_w = st.slider("Trọng số quán tính trong fusion", 0.0, 1.0,
                             float(enr.fusion_w), 0.05)
else:
    fusion_w = None


# ── Main ──
if enr is None:
    st.info("⬅️ Chọn chủ máy và bấm **Đăng ký** để bắt đầu.")
    st.stop()

users = st.session_state["users"]
data_dir = Path(st.session_state["dir"])
others = [u for u in users if u != enr.owner_id]
has_touch = enr.rf_touch is not None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Chủ máy", enr.owner_id)
c2.metric("Ngưỡng quán tính", f"{enr.thr_inertial:.2f}")
c3.metric("Ngưỡng fusion", f"{enr.thr_fusion:.2f}" if has_touch else "—")
c4.metric("Ngữ cảnh", MODE_LABEL[enr.data_mode])

if not has_touch:
    st.warning("Không có dữ liệu touch cho chủ máy này → chỉ hiển thị nhánh quán tính. "
               "Cột touch/fusion sẽ trống.")

st.divider()
st.subheader("Bảng chấm điểm: Chủ máy vs Người lạ")
st.caption("Hai cột kết luận để đối chiếu: **theo quán tính** (bản triển khai) và "
           "**theo fusion**. Cột 'Đúng?' xanh nếu kết luận khớp thực tế.")

max_test_sessions = st.slider(
    "Số phiên test tối đa mỗi người lạ", 1, 6, 6,
    help="Tăng để chấm nhiều phiên của mỗi người lạ → nhiều mẫu hơn, "
         "thống kê đáng tin hơn. Mặc định lấy tất cả (tối đa 6).")

if st.button("▶️ Chấm điểm tất cả", type="primary"):
    test_candidates = [s for s in sorted(load_user_inertial(enr.owner_id, data_dir, mode=enr.data_mode))
                       if s not in enr.enroll_sessions]
    if not test_candidates:
        test_candidates = sorted(load_user_inertial(enr.owner_id, data_dir, mode=enr.data_mode))[-1:]

    rows = []
    # chủ máy
    for s in test_candidates:
        r = verify_session(enr, enr.owner_id, s, data_dir, encoder, fusion_w_override=fusion_w)
        if r["p_inertial"] is not None:
            rows.append(r)
    # người lạ (tất cả session mỗi người)
    for u in others:
        sess = sorted(load_user_inertial(u, data_dir, mode=enr.data_mode))
        for sid in sess[:max_test_sessions]:
            r = verify_session(enr, u, sid, data_dir, encoder, fusion_w_override=fusion_w)
            if r["p_inertial"] is not None:
                rows.append(r)

    if rows:
        df = pd.DataFrame([{
            "Người": (r["test_user"] + " (chủ)" if r["is_actual_owner"] else r["test_user"]),
            "Session": r["session"],
            "Điểm quán tính": round(r["p_inertial"], 3),
            "Điểm touch": fmt(r["p_touch"]),
            "Điểm fusion": round(r["p_fusion"], 3) if r["p_fusion"] is not None else "—",
            "Kết luận (quán tính)": r["decision_inertial"],
            "Kết luận (fusion)": r["decision_fusion"],
            "Đúng? (quán tính)": "✓" if r["correct_inertial"] else "✗",
            "Đúng? (fusion)": "✓" if r["correct_fusion"] else "✗",
        } for r in rows])

        st.dataframe(df, use_container_width=True, hide_index=True)

        acc_in = np.mean([r["correct_inertial"] for r in rows]) * 100
        acc_fu = np.mean([r["correct_fusion"] for r in rows]) * 100
        m1, m2 = st.columns(2)
        m1.metric("Độ chính xác — quán tính (triển khai)", f"{acc_in:.0f}%")
        m2.metric("Độ chính xác — fusion", f"{acc_fu:.0f}%" if has_touch else "—")
        if has_touch and acc_fu <= acc_in:
            st.info("Fusion **không vượt** nhánh quán tính — minh họa đúng kết luận của báo cáo: "
                    "trong điều kiện dữ liệu này, dung hợp touch không cải thiện open-set.")

st.divider()
with st.expander("ℹ️ Vì sao bản triển khai trên app là đơn modal?"):
    st.markdown(
        "- **Về mô hình**, hệ thống hỗ trợ đa phương thức và web demo này trình bày đầy đủ "
        "điểm quán tính, touch và fusion.\n"
        "- **Trên Android**, thu sự kiện chạm ở chế độ nền bị giới hạn bởi quyền hệ thống "
        "(Accessibility), nên bản on-device dùng cấu hình đơn modal quán tính.\n"
        "- Ngoài ra, trong đánh giá open-set, **fusion touch không cải thiện** so với nhánh "
        "quán tính — web demo cho thấy trực tiếp điều này qua hai cột kết luận."
    )
    