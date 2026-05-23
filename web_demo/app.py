"""
app.py — Active Auth Verification Demo (Streamlit)

Usage:
    streamlit run app.py

Configure paths in the sidebar at first run.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch

from models import load_encoder
from verifier import (
    list_available_users, load_user_inertial,
    enroll, verify_session, verify_user_sessions, verify_batch_impostors,
    load_artifacts,
)


# ═══════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Active Auth — Verification Demo",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔐 Active Authentication — Verification Demo")
st.caption(
    "Demo khoa học cho thesis defense — Reproduce training methodology V5: "
    "Random Forest per-user trên embedding (inertial) + 48-D touch features, "
    "score-level fusion. Sử dụng pre-built impostor pool + scaler từ training."
)


# ═══════════════════════════════════════════════════════════════════
# Path config — cached resources
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading backbone CNN...")
def load_backbone_cached(checkpoint_path: str, n_users: int):
    return load_encoder(checkpoint_path, n_users=n_users)


with st.sidebar:
    st.header("⚙️ Cấu hình")
    data_dir_str = st.text_input("processed_data path",
                                 value="processed_data",
                                 help="Folder chứa thư mục userX với X.npy + touch_session_features.csv")
    newbie_dir_str = st.text_input("newbie_data path",
                                   value="newbie_data",
                                   help="Folder chứa user CHƯA TỪNG có trong training (unseen)")
    model_path_str = st.text_input("backbone.pt path",
                                   value="models/backbone.pt")
    export_dir_str = st.text_input("export/ path",
                                   value="export",
                                   help="Folder chứa impostor_pool_*.npy + touch_scaler.json từ training V5")

    data_dir = Path(data_dir_str)
    newbie_dir = Path(newbie_dir_str)
    model_path = Path(model_path_str)
    export_dir = Path(export_dir_str)

    if not data_dir.exists():
        st.error(f"❌ Không tìm thấy: {data_dir}")
        st.stop()
    if not model_path.exists():
        st.error(f"❌ Không tìm thấy: {model_path}")
        st.stop()
    if not export_dir.exists():
        st.error(f"❌ Không tìm thấy: {export_dir} (cần pool + scaler từ training)")
        st.stop()

    users = list_available_users(data_dir)
    if len(users) < 2:
        st.error("Cần ≥ 2 users trong processed_data/")
        st.stop()

    st.success(f"✓ {len(users)} cohort users found")

    # Check for newbie data (optional — Tab 4 will show warning if empty)
    newbie_users = list_available_users(newbie_dir) if newbie_dir.exists() else []
    if newbie_users:
        st.success(f"✓ {len(newbie_users)} newbie users found")
    else:
        st.caption(f"○ Không có newbie data tại `{newbie_dir}` (Tab 4 sẽ disabled)")

    # Load encoder (cached)
    try:
        encoder = load_backbone_cached(str(model_path), n_users=len(users))
    except Exception as e:
        st.error(f"Lỗi load backbone: {e}")
        st.stop()

    # Load V5 artifacts (impostor pool + scaler) — cached
    @st.cache_resource(show_spinner="Loading V5 artifacts...")
    def load_artifacts_cached(export_dir_str: str):
        return load_artifacts(Path(export_dir_str))

    try:
        artifacts = load_artifacts_cached(str(export_dir))
        st.success(f"✓ V5 artifacts loaded: pool_i {artifacts.pool_inertial.shape}, "
                   f"pool_t {artifacts.pool_touch_scaled.shape}")
    except Exception as e:
        st.error(f"Lỗi load artifacts: {e}")
        st.stop()

    st.divider()
    st.header("👤 Enrollment")

    # Owner pool selection — cho phép enroll newbie làm owner
    pool_options = ["Cohort (đã trong training)"]
    if newbie_users:
        pool_options.append("Newbie (deploy thực tế — chưa trong training)")

    owner_pool = st.radio(
        "Owner pool",
        pool_options,
        index=0,
        help=("Cohort = test in-distribution. "
              "Newbie = mô phỏng deploy thực tế: app cho người mới mua máy."),
    )

    if owner_pool.startswith("Cohort"):
        available_owners = users
        owner_dir = data_dir
        pool_label = "cohort"
    else:
        available_owners = newbie_users
        owner_dir = newbie_dir
        pool_label = "newbie"

    owner_id = st.selectbox("Owner user", available_owners, index=0,
                            key=f"owner_select_{pool_label}")

    # Determine max sessions available
    try:
        sessions = load_user_inertial(owner_id, owner_dir)
        n_total_sessions = len(sessions)
    except Exception as e:
        st.error(f"Lỗi load {owner_id}: {e}")
        st.stop()

    n_enroll = st.slider(
        "Số session để enroll",
        min_value=1,
        max_value=max(1, n_total_sessions - 1),
        value=min(4, max(1, n_total_sessions - 1)),
        help="Còn lại sẽ dùng làm test data của chính owner",
    )

    if st.button("🎯 Enroll", type="primary", use_container_width=True):
        with st.spinner(f"Training per-user RF cho {owner_id}..."):
            try:
                enrollment = enroll(owner_id, n_enroll, owner_dir, encoder, artifacts)
                st.session_state['enrollment'] = enrollment
                st.session_state['n_enroll'] = n_enroll
                st.session_state['n_total_sessions'] = n_total_sessions
                st.session_state['owner_pool'] = pool_label
                st.session_state['owner_dir'] = str(owner_dir)
                # Clear previous results
                for key in ['last_own_results', 'last_imp_results',
                            'last_batch_results', 'last_newbie_results']:
                    st.session_state.pop(key, None)
                st.success(f"✓ Enrolled {owner_id} ({pool_label}) với "
                           f"{n_enroll}/{n_total_sessions} sessions "
                           f"(fusion_w = {enrollment.fusion_w:.2f})")
            except Exception as e:
                st.error(f"Lỗi enrollment: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()
    st.header("⚖️ Threshold")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.01,
        help="Tăng → reject nhiều hơn (FAR ↓, FRR ↑). Giảm → accept dễ hơn (FAR ↑, FRR ↓).",
    )
    st.session_state['threshold'] = threshold

    if 'enrollment' in st.session_state:
        e = st.session_state['enrollment']
        pool_emoji = "🆕" if st.session_state.get('owner_pool') == 'newbie' else "📚"
        st.info(
            f"**Active**: {pool_emoji} {e.owner_id}\n\n"
            f"Sessions: {', '.join(e.enroll_sessions)}\n\n"
            f"RF inertial: ✓ {e.rf_inertial.n_estimators} trees\n\n"
            f"RF touch: {'✓ trained' if e.rf_touch else '✗ skipped (insufficient touch data)'}"
        )


# ═══════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════

if 'enrollment' not in st.session_state:
    st.info("👈 Pick owner và click **Enroll** trong sidebar để bắt đầu.")
    st.stop()

enrollment = st.session_state['enrollment']
n_enroll = st.session_state['n_enroll']
n_total_sessions = st.session_state['n_total_sessions']
threshold = st.session_state.get('threshold', 0.50)


def apply_threshold(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Re-apply threshold lên cột 'fused' (hoặc 'p_inertial' nếu không có fused)."""
    df = df.copy()
    score_col = 'fused' if 'fused' in df.columns else 'p_inertial'
    df['decision'] = df[score_col].apply(
        lambda x: 'TRUSTED' if pd.notna(x) and float(x) >= thr else 'REJECTED'
    )
    if 'is_actual_owner' in df.columns:
        df['correct'] = df.apply(
            lambda r: (r['decision'] == 'TRUSTED') if r['is_actual_owner']
                      else (r['decision'] == 'REJECTED'),
            axis=1,
        )
    return df


# Helper: render result table
def render_results_table(df: pd.DataFrame, height: int = 300):
    """Display verification results with color-coded decisions."""
    if df.empty:
        st.warning("No results to display")
        return

    display = df.copy()
    # Format probabilities
    for col in ['p_inertial', 'p_touch', 'fused']:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    # Add colored decision
    def color_decision(val):
        if val == 'TRUSTED':
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif val == 'REJECTED':
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        return ''

    styled = display.style.map(color_decision, subset=['decision'])
    st.dataframe(styled, use_container_width=True, height=height)


def render_score_breakdown_chart(df: pd.DataFrame):
    """Bar chart showing p_inertial vs p_touch vs fused for each row."""
    if df.empty:
        return

    # Long format for grouped bar
    plot_data = []
    for _, row in df.iterrows():
        label = f"{row['test_user']} / {row['session']}"
        for metric in ['p_inertial', 'p_touch', 'fused']:
            val = row.get(metric)
            if pd.notna(val):
                plot_data.append({
                    'sample': label,
                    'modality': metric.replace('p_', '').replace('_', ' '),
                    'score': float(val),
                })

    if not plot_data:
        return

    df_plot = pd.DataFrame(plot_data)
    fig = px.bar(
        df_plot,
        x='sample', y='score', color='modality',
        barmode='group',
        title='Score breakdown by modality',
        labels={'score': 'P(owner)', 'sample': 'Test sample'},
        color_discrete_map={'inertial': '#4C72B0', 'touch': '#DD8452', 'fused': '#55A868'},
        range_y=[0, 1],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Decision threshold (0.5)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "✅ Own data (TRUSTED expected)",
    "❌ Single impostor (in-cohort)",
    "📊 Batch on ALL impostors (in-cohort)",
    "🆕 Newbie test (UNSEEN users)",
])


# ─── TAB 1: Own data ──────────────────────────────────────────────
with tab1:
    st.subheader(f"Test trên dữ liệu KHÁC của chính {enrollment.owner_id}")

    # Owner có thể là cohort hoặc newbie — dùng dir được lưu khi enroll
    owner_dir_for_test = Path(st.session_state.get('owner_dir', str(data_dir)))
    own_sessions = sorted(load_user_inertial(enrollment.owner_id, owner_dir_for_test).keys())
    test_sessions = [s for s in own_sessions if s not in enrollment.enroll_sessions]

    if not test_sessions:
        st.warning(f"{enrollment.owner_id} đã dùng hết sessions cho enrollment. "
                  f"Giảm 'Số session để enroll' trong sidebar.")
    else:
        st.write(f"Enrollment: `{', '.join(enrollment.enroll_sessions)}` "
                f"(N={len(enrollment.enroll_sessions)})")
        st.write(f"Test on:    `{', '.join(test_sessions)}` "
                f"(N={len(test_sessions)})")

        if st.button("▶ Run own-data verification", key="run_own", type="primary"):
            with st.spinner("Verifying..."):
                df = verify_user_sessions(enrollment, enrollment.owner_id,
                                          test_sessions, owner_dir_for_test, encoder)
                st.session_state['last_own_results'] = df

        if 'last_own_results' in st.session_state:
            df = apply_threshold(st.session_state['last_own_results'], threshold)

            # Metrics
            n_total = len(df)
            n_trusted = (df['decision'] == 'TRUSTED').sum()
            frr = (df['decision'] == 'REJECTED').sum() / max(n_total, 1)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Sessions tested", n_total)
            col_b.metric("Accepted (TRUSTED)", f"{n_trusted}/{n_total}")
            col_c.metric("FRR (False Reject Rate)", f"{frr*100:.1f}%",
                        delta=f"{(0.05-frr)*100:.1f}pp vs 5% target",
                        delta_color="normal")

            render_results_table(df)
            render_score_breakdown_chart(df)


# ─── TAB 2: Single impostor ───────────────────────────────────────
with tab2:
    st.subheader(f"Test 1 user khác (impostor) — phải bị REJECTED")

    other_users = [u for u in users if u != enrollment.owner_id]
    impostor_id = st.selectbox("Pick impostor user", other_users, key="imp_sel")

    imp_sessions = sorted(load_user_inertial(impostor_id, data_dir).keys())
    n_imp = st.slider("Số session of impostor để test", 1,
                     min(4, len(imp_sessions)), min(2, len(imp_sessions)),
                     key="imp_n")

    test_imp_sessions = imp_sessions[:n_imp]

    if st.button("▶ Run impostor verification", key="run_imp", type="primary"):
        with st.spinner("Verifying..."):
            df = verify_user_sessions(enrollment, impostor_id,
                                      test_imp_sessions, data_dir, encoder)
            st.session_state['last_imp_results'] = df

    if 'last_imp_results' in st.session_state:
        df = apply_threshold(st.session_state['last_imp_results'], threshold)

        n_total = len(df)
        n_rejected = (df['decision'] == 'REJECTED').sum()
        far = (df['decision'] == 'TRUSTED').sum() / max(n_total, 1)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Sessions tested", n_total)
        col_b.metric("Rejected (impostor)", f"{n_rejected}/{n_total}")
        col_c.metric("FAR (False Accept Rate)", f"{far*100:.1f}%",
                    delta=f"{(0.05-far)*100:.1f}pp vs 5% target",
                    delta_color="normal")

        render_results_table(df)
        render_score_breakdown_chart(df)


# ─── TAB 3: Batch on all impostors ────────────────────────────────
with tab3:
    st.subheader(f"Test trên TẤT CẢ {len(users)-1} users khác — tính FAR thực tế")

    n_per_user = st.slider("Số session/user khi test", 1, 3, 1, key="batch_n")

    if st.button("▶ Run batch verification", key="run_batch", type="primary"):
        with st.spinner(f"Verifying {(len(users)-1) * n_per_user} sessions..."):
            df = verify_batch_impostors(enrollment, data_dir, encoder,
                                        n_sessions_per_user=n_per_user)
            st.session_state['last_batch_results'] = df

    if 'last_batch_results' in st.session_state:
        df = apply_threshold(st.session_state['last_batch_results'], threshold)

        n_total = len(df)
        n_false_accept = (df['decision'] == 'TRUSTED').sum()
        far = n_false_accept / max(n_total, 1)

        # Also include own-data results if available, for FRR
        own_df = (apply_threshold(st.session_state['last_own_results'], threshold)
                  if 'last_own_results' in st.session_state else None)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Impostor sessions", n_total)
        col_b.metric("False accepts", n_false_accept)
        col_c.metric("FAR", f"{far*100:.2f}%",
                    delta=f"{(0.05-far)*100:.2f}pp vs 5%",
                    delta_color="normal")
        if own_df is not None:
            frr_local = (own_df['decision'] == 'REJECTED').sum() / max(len(own_df), 1)
            col_d.metric("FRR (from Tab 1)", f"{frr_local*100:.2f}%")
        else:
            col_d.info("Run Tab 1 to get FRR")

        # Distribution chart: fused scores
        if 'fused' in df.columns and df['fused'].notna().any():
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['fused'].dropna(),
                nbinsx=20,
                name='Impostor fused scores',
                marker_color='#DD8452',
                opacity=0.75,
            ))
            if own_df is not None:
                fig.add_trace(go.Histogram(
                    x=own_df['fused'].dropna(),
                    nbinsx=20,
                    name='Owner fused scores',
                    marker_color='#55A868',
                    opacity=0.75,
                ))
            fig.update_layout(
                title="Fused score distribution: owner (green) vs impostors (orange)",
                xaxis_title="Fused P(owner)",
                yaxis_title="# sessions",
                barmode='overlay',
                height=400,
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                         annotation_text="Threshold 0.5")
            st.plotly_chart(fig, use_container_width=True)

        # Sort by fused score
        df_sorted = df.sort_values('fused', ascending=False).reset_index(drop=True)
        render_results_table(df_sorted, height=500)

        # Top false-accepts
        false_accepts = df_sorted[df_sorted['decision'] == 'TRUSTED']
        if len(false_accepts) > 0:
            st.warning(f"⚠️ {len(false_accepts)} false accepts:")
            st.dataframe(false_accepts[['test_user', 'session', 'fused',
                                       'p_inertial', 'p_touch']],
                        use_container_width=True)


# ─── TAB 4: Newbie (unseen) users ─────────────────────────────────
with tab4:
    st.subheader("Test trên users CHƯA TỪNG có trong training cohort")
    st.markdown(
        "**Đây là kịch bản quan trọng nhất** cho thesis defense — bằng chứng "
        "model generalize được sang user lạ, không chỉ phân biệt giữa các user "
        "đã thấy trong training. Backbone CNN không học identity của các user "
        "này, chỉ trích đặc trưng general từ inertial signals."
    )

    if not newbie_users:
        st.warning(
            f"Không tìm thấy newbie users tại `{newbie_dir}`. "
            f"Đặt data theo cùng cấu trúc với `processed_data/`:\n\n"
            f"```\nnewbie_data/\n├── newbie1/\n│   ├── X.npy\n│   ├── session.npy\n"
            f"│   ├── touch_tap.csv\n│   ├── touch_scroll.csv\n│   └── touch_key.csv\n"
            f"└── newbie2/\n    └── ...\n```"
        )
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            mode = st.radio(
                "Test mode",
                ["Single newbie (chi tiết)", "All newbies (batch)"],
                key="newbie_mode",
                horizontal=True,
            )
        with col_b:
            n_sess_newbie = st.slider("Sessions/newbie", 1, 5, 2, key="newbie_n_sess")

        if mode == "Single newbie (chi tiết)":
            picked = st.selectbox("Pick newbie", newbie_users, key="newbie_pick")
            if st.button("▶ Run newbie test", key="run_newbie_single", type="primary"):
                with st.spinner(f"Testing {picked}..."):
                    sessions_dict = load_user_inertial(picked, newbie_dir)
                    test_sess = sorted(sessions_dict.keys())[:n_sess_newbie]
                    df = verify_user_sessions(enrollment, picked, test_sess,
                                              newbie_dir, encoder)
                    st.session_state['last_newbie_results'] = df
        else:
            if st.button(f"▶ Batch test trên TẤT CẢ {len(newbie_users)} newbies",
                        key="run_newbie_batch", type="primary"):
                with st.spinner(f"Testing {len(newbie_users)} newbies..."):
                    all_rows = []
                    for nb in newbie_users:
                        sessions_dict = load_user_inertial(nb, newbie_dir)
                        keys = sorted(sessions_dict.keys())[:n_sess_newbie]
                        for s in keys:
                            row = verify_session(enrollment, nb, s, newbie_dir, encoder)
                            all_rows.append(row)
                    df = pd.DataFrame(all_rows)
                    st.session_state['last_newbie_results'] = df

        if 'last_newbie_results' in st.session_state:
            df = apply_threshold(st.session_state['last_newbie_results'], threshold)
            n_total = len(df)
            n_false_accept = (df['decision'] == 'TRUSTED').sum()
            far_newbie = n_false_accept / max(n_total, 1)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Newbie sessions tested", n_total)
            col_b.metric("False accepts (TRUSTED)", n_false_accept)
            col_c.metric("FAR on UNSEEN users", f"{far_newbie*100:.2f}%",
                        delta=f"{(0.05-far_newbie)*100:.2f}pp vs 5% target",
                        delta_color="normal")

            if far_newbie == 0:
                st.success(
                    "🎉 **0% FAR trên newbie** — model generalize hoàn toàn. "
                    "Tất cả users chưa từng thấy đều bị reject đúng."
                )
            elif far_newbie < 0.05:
                st.success(
                    f"✓ FAR trên newbie = {far_newbie*100:.2f}% — dưới target 5%. "
                    "Model generalize tốt sang user unseen."
                )
            else:
                st.warning(
                    f"⚠ FAR trên newbie = {far_newbie*100:.2f}% — vượt 5%. "
                    "Generalization ra ngoài cohort có hạn chế."
                )

            # Score distribution chart
            if 'fused' in df.columns and df['fused'].notna().any():
                own_df = st.session_state.get('last_own_results')
                imp_df = st.session_state.get('last_batch_results')

                fig = go.Figure()
                if own_df is not None and 'fused' in own_df.columns:
                    fig.add_trace(go.Histogram(
                        x=own_df['fused'].dropna(),
                        nbinsx=20, name='Owner (in-cohort)',
                        marker_color='#55A868', opacity=0.7,
                    ))
                if imp_df is not None and 'fused' in imp_df.columns:
                    fig.add_trace(go.Histogram(
                        x=imp_df['fused'].dropna(),
                        nbinsx=20, name='Impostor (in-cohort)',
                        marker_color='#DD8452', opacity=0.5,
                    ))
                fig.add_trace(go.Histogram(
                    x=df['fused'].dropna(),
                    nbinsx=20, name='Newbie (UNSEEN)',
                    marker_color='#C44E52', opacity=0.9,
                ))
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                              annotation_text="Threshold 0.5")
                fig.update_layout(
                    title="Score distribution: Owner vs In-cohort Impostor vs UNSEEN Newbie",
                    xaxis_title="Fused P(owner)",
                    yaxis_title="# sessions",
                    barmode='overlay',
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

            render_results_table(df.sort_values('fused', ascending=False)
                                   .reset_index(drop=True),
                                height=400)

            false_accepts = df[df['decision'] == 'TRUSTED']
            if len(false_accepts) > 0:
                st.warning(f"⚠ {len(false_accepts)} false accepts trên newbies:")
                st.dataframe(false_accepts[['test_user', 'session', 'fused',
                                           'p_inertial', 'p_touch']],
                            use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Methodology: Train per-user RF (inertial 128-D embedding + touch 27-D), "
    "fuse với w=0.5, decision threshold = 0.5. "
    "Impostor pool built on-the-fly từ non-owner users (no leakage). "
    "Tương ứng với training pipeline Active_Auth V5."
)