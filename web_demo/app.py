"""
app.py — Active Auth Verification Demo (Streamlit)

Demo so sánh 6 biến thể model = 3 kiến trúc × 2 chế độ training:
    cnn / convlstm / convlstm_bi   ×   walking (1 action) / all (3 actions)

Mỗi lần verify 1 session, app chạy qua TẤT CẢ 6 biến thể và hiển thị kết quả
song song để so sánh.

Usage:
    streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from models import load_encoder
from verifier import (
    list_available_users, load_user_inertial,
    enroll, verify_session, load_artifacts,
)


# ═══════════════════════════════════════════════════════════════════
# Variant config
# ═══════════════════════════════════════════════════════════════════

# Mỗi variant = (arch, data_mode). Tổng cộng 6 biến thể.
ARCHITECTURES = ['cnn', 'convlstm', 'convlstm_bi']
DATA_MODES    = ['walking', 'all']   # walking = 1 action, all = 3 actions

# Map arch → folder trong artifacts root
ARCH_DIR = {
    'cnn':         'cnn_v2',
    'convlstm':    'convlstm_v2',
    'convlstm_bi': 'convlstm_bi_v2',
}

# Nhãn hiển thị cho data_mode
MODE_LABEL = {
    'walking': '1 action (walking)',
    'all':     '3 actions (all)',
}

# Tên đẹp cho từng kiến trúc
ARCH_LABEL = {
    'cnn':         'CNN',
    'convlstm':    'ConvLSTM',
    'convlstm_bi': 'ConvLSTM-Bi',
}


def variant_key(arch: str, mode: str) -> str:
    return f"{arch}__{mode}"


def variant_paths(artifacts_root: Path, arch: str, mode: str) -> dict:
    arch_dir = artifacts_root / ARCH_DIR[arch]
    return {
        'backbone': arch_dir / f'models_{mode}' / 'backbone.pt',
        'export':   arch_dir / f'export_{mode}',
    }


ALL_VARIANTS = [(a, m) for a in ARCHITECTURES for m in DATA_MODES]


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
    "So sánh song song 6 biến thể model = 3 kiến trúc (CNN / ConvLSTM / ConvLSTM-Bi) "
    "× 2 chế độ training (1 action walking / 3 actions all). "
    "Mỗi session test được score qua tất cả 6 model để đối chiếu trực tiếp."
)


# ═══════════════════════════════════════════════════════════════════
# Cached resource loaders
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_backbone_cached(checkpoint_path: str, arch: str, n_users: int):
    return load_encoder(checkpoint_path, n_users=n_users, arch=arch)


@st.cache_resource(show_spinner=False)
def load_artifacts_cached(export_dir_str: str):
    return load_artifacts(Path(export_dir_str))


# ═══════════════════════════════════════════════════════════════════
# Sidebar — config + enrollment
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Cấu hình")

    data_dir_str = st.text_input(
        "processed_data path", value="processed_data",
        help="Folder cohort users (đã có trong training)",
    )
    newbie_dir_str = st.text_input(
        "newbie_data path", value="newbie_data",
        help="Folder user UNSEEN (chưa từng có trong training)",
    )
    artifacts_root_str = st.text_input(
        "artifacts root", value="artifacts",
        help=("Folder gốc chứa 3 sub-folder: cnn_v2/, convlstm_v2/, convlstm_bi_v2/. "
              "Mỗi sub-folder phải có models_walking/, models_all/, "
              "export_walking/, export_all/."),
    )

    data_dir = Path(data_dir_str)
    newbie_dir = Path(newbie_dir_str)
    artifacts_root = Path(artifacts_root_str)

    if not data_dir.exists():
        st.error(f"❌ Không tìm thấy: {data_dir}")
        st.stop()
    if not artifacts_root.exists():
        st.error(f"❌ Không tìm thấy artifacts root: {artifacts_root}")
        st.stop()

    # Validate tất cả 6 variant paths có tồn tại không
    missing = []
    for arch, mode in ALL_VARIANTS:
        paths = variant_paths(artifacts_root, arch, mode)
        if not paths['backbone'].exists():
            missing.append(f"{paths['backbone']}")
        if not paths['export'].exists():
            missing.append(f"{paths['export']}")
    if missing:
        st.error("❌ Thiếu các file/folder sau:\n" + "\n".join(f"- `{p}`" for p in missing[:6]))
        st.stop()

    users = list_available_users(data_dir)
    if len(users) < 2:
        st.error("Cần ≥ 2 users trong processed_data/")
        st.stop()

    st.success(f"✓ {len(users)} cohort users found")

    newbie_users = list_available_users(newbie_dir) if newbie_dir.exists() else []
    if newbie_users:
        st.success(f"✓ {len(newbie_users)} newbie users found")
    else:
        st.caption(f"○ Không có newbie data tại `{newbie_dir}` (Tab 4 disabled)")

    # ── Load 6 encoders + 6 artifacts (cached) ────────────────────
    with st.spinner("Loading 6 model variants..."):
        encoders   = {}
        artifacts_map = {}
        load_errors = []
        for arch, mode in ALL_VARIANTS:
            key = variant_key(arch, mode)
            paths = variant_paths(artifacts_root, arch, mode)
            try:
                encoders[key] = load_backbone_cached(
                    str(paths['backbone']), arch=arch, n_users=len(users),
                )
                artifacts_map[key] = load_artifacts_cached(str(paths['export']))
            except Exception as e:
                load_errors.append(f"{key}: {e}")

    if load_errors:
        st.error("Lỗi load:\n" + "\n".join(load_errors))
        st.stop()

    st.success(f"✓ Loaded {len(encoders)}/6 variants")

    st.divider()
    st.header("👤 Enrollment")

    # Owner pool (cohort vs newbie)
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

    # Số session tối đa dựa trên union của 2 mode
    try:
        sessions_all = load_user_inertial(owner_id, owner_dir, mode='all')
        sessions_walking = load_user_inertial(owner_id, owner_dir, mode='walking')
        common_sessions = set(sessions_all.keys()) & set(sessions_walking.keys())
        n_total_sessions = len(common_sessions) if common_sessions else max(
            len(sessions_all), len(sessions_walking),
        )
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

    if st.button("🎯 Enroll all 6 variants", type="primary", use_container_width=True):
        # Khi owner là newbie, pool âm tính phải đến từ COHORT, không phải
        # newbie_dir — nếu không RF sẽ học sai phân phối.
        impostor_dir_for_enroll = (
            data_dir if pool_label == 'newbie' else owner_dir
        )

        enrollments = {}
        progress = st.progress(0.0, text="Enrolling...")
        for i, (arch, mode) in enumerate(ALL_VARIANTS):
            key = variant_key(arch, mode)
            label = f"{ARCH_LABEL[arch]} × {MODE_LABEL[mode]}"
            progress.progress((i) / len(ALL_VARIANTS),
                              text=f"Training RF: {label}...")
            try:
                enr = enroll(
                    owner_id, n_enroll, owner_dir,
                    encoders[key], artifacts_map[key],
                    data_mode=mode,
                    impostor_dir=impostor_dir_for_enroll,
                )
                enrollments[key] = enr
            except Exception as e:
                st.warning(f"⚠ Variant `{key}` enroll thất bại: {e}")

        progress.progress(1.0, text="Done.")

        if not enrollments:
            st.error("Tất cả variant đều enroll thất bại.")
            st.stop()

        st.session_state['enrollments'] = enrollments
        st.session_state['n_enroll'] = n_enroll
        st.session_state['n_total_sessions'] = n_total_sessions
        st.session_state['owner_pool'] = pool_label
        st.session_state['owner_dir'] = str(owner_dir)
        st.session_state['owner_id'] = owner_id

        # Clear previous results
        for k in ['last_own_results', 'last_imp_results',
                  'last_batch_results', 'last_newbie_results']:
            st.session_state.pop(k, None)

        st.success(f"✓ Enrolled `{owner_id}` ({pool_label}) trên {len(enrollments)}/6 variants")

    st.divider()
    st.header("⚖️ Threshold")

    enrolled = st.session_state.get('enrollments')

    use_adaptive = st.toggle(
        "Dùng adaptive threshold per-variant (EER)",
        value=True,
        help=("Bật: mỗi variant dùng threshold EER riêng (tính từ val set lúc enroll). "
              "Tắt: dùng 1 threshold thủ công áp dụng cho cả 6 variant."),
    )

    if use_adaptive:
        threshold_override = None
        if enrolled:
            thrs = [e.adaptive_threshold for e in enrolled.values()]
            st.caption(
                f"Threshold per-variant: min={min(thrs):.3f}, max={max(thrs):.3f}, "
                f"mean={np.mean(thrs):.3f}"
            )
        else:
            st.caption("Enroll trước để tính adaptive threshold.")
    else:
        default_thr = 0.50
        if enrolled:
            default_thr = float(round(np.mean([e.adaptive_threshold
                                               for e in enrolled.values()]), 2))
        threshold_override = st.slider(
            "Threshold manual (áp cho cả 6 variant)",
            min_value=0.10, max_value=0.90,
            value=default_thr, step=0.01,
            help="Tăng → reject nhiều hơn (FAR ↓, FRR ↑). Giảm → ngược lại.",
        )

    st.session_state['threshold_override'] = threshold_override

    if enrolled:
        e_first = next(iter(enrolled.values()))
        pool_emoji = "🆕" if st.session_state.get('owner_pool') == 'newbie' else "📚"
        st.info(
            f"**Active**: {pool_emoji} {e_first.owner_id}\n\n"
            f"Sessions: {', '.join(e_first.enroll_sessions)}\n\n"
            f"Variants enrolled: {len(enrolled)}/6"
        )


# ═══════════════════════════════════════════════════════════════════
# Main area — guard
# ═══════════════════════════════════════════════════════════════════

if 'enrollments' not in st.session_state:
    st.info("👈 Pick owner và click **Enroll** trong sidebar để bắt đầu.")
    st.stop()

enrollments: dict = st.session_state['enrollments']
n_enroll = st.session_state['n_enroll']
n_total_sessions = st.session_state['n_total_sessions']
threshold_override = st.session_state.get('threshold_override', None)
owner_id_active = st.session_state['owner_id']


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_threshold_for(variant_k: str) -> float:
    """Lấy threshold cho 1 variant. Nếu có override → dùng nó, else adaptive."""
    if threshold_override is not None:
        return float(threshold_override)
    return float(enrollments[variant_k].adaptive_threshold)


def apply_thresholds_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply threshold cho mỗi row dựa trên cột 'variant'."""
    if df.empty:
        return df
    df = df.copy()
    score_col = 'fused' if 'fused' in df.columns else 'p_inertial'

    def decide(row):
        thr = get_threshold_for(row['variant'])
        val = row.get(score_col)
        if pd.isna(val):
            return 'NO_DATA', thr
        return ('TRUSTED' if float(val) >= thr else 'REJECTED'), thr

    decisions, thrs = zip(*[decide(r) for _, r in df.iterrows()])
    df['decision'] = decisions
    df['threshold'] = thrs

    if 'is_actual_owner' in df.columns:
        df['correct'] = df.apply(
            lambda r: (r['decision'] == 'TRUSTED') if r['is_actual_owner']
                      else (r['decision'] == 'REJECTED'),
            axis=1,
        )
    return df


def verify_session_all_variants(test_user: str, session_id: str,
                                 src_dir: Path) -> list:
    """Chạy verify_session qua TẤT CẢ variants đã enroll. Trả list[dict]."""
    rows = []
    for key, enr in enrollments.items():
        encoder = encoders[key]
        try:
            row = verify_session(enr, test_user, session_id, src_dir, encoder)
        except Exception as e:
            row = {
                'test_user': test_user, 'session': session_id,
                'p_inertial': None, 'p_touch': None, 'fused': None,
                'decision': 'ERROR', 'n_windows': 0,
                'is_actual_owner': (test_user == enr.owner_id),
                'correct': False,
            }
        row['variant'] = key
        arch, mode = key.split('__')
        row['model'] = ARCH_LABEL[arch]
        row['train_mode'] = MODE_LABEL[mode]
        row['fusion_w'] = enr.fusion_w
        rows.append(row)
    return rows


def verify_user_sessions_all_variants(test_user: str, session_ids: list,
                                       src_dir: Path) -> pd.DataFrame:
    rows = []
    for s in session_ids:
        rows.extend(verify_session_all_variants(test_user, s, src_dir))
    return pd.DataFrame(rows)


def verify_batch_impostors_all_variants(other_users: list, n_per_user: int,
                                         src_dir: Path) -> pd.DataFrame:
    """Test ALL variants × ALL impostor sessions."""
    rows = []
    for u in other_users:
        # Session keys từ union của 2 mode để cover được data của cả variants
        sess_w = load_user_inertial(u, src_dir, mode='walking')
        sess_a = load_user_inertial(u, src_dir, mode='all')
        common = sorted(set(sess_w.keys()) | set(sess_a.keys()))
        keys = common[:n_per_user]
        for s in keys:
            rows.extend(verify_session_all_variants(u, s, src_dir))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════

DISPLAY_COLS = [
    'model', 'train_mode', 'test_user', 'session',
    'p_inertial', 'p_touch', 'fused', 'threshold', 'decision', 'n_windows',
]


def render_results_table(df: pd.DataFrame, height: int = 420):
    """Hiển thị bảng kết quả long-form với color-coded decision."""
    if df.empty:
        st.warning("No results to display.")
        return

    display = df.copy()

    # Sort: test_user, session, model, train_mode
    sort_cols = [c for c in ['test_user', 'session', 'model', 'train_mode']
                 if c in display.columns]
    display = display.sort_values(sort_cols).reset_index(drop=True)

    # Format probabilities
    for col in ['p_inertial', 'p_touch', 'fused', 'threshold']:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )

    # Giữ thứ tự cột để hiển thị
    cols_show = [c for c in DISPLAY_COLS if c in display.columns]
    display = display[cols_show]

    def color_decision(val):
        if val == 'TRUSTED':
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        if val == 'REJECTED':
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        if val in ('NO_DATA', 'ERROR'):
            return 'background-color: #fff3cd; color: #856404;'
        return ''

    styled = display.style.map(color_decision, subset=['decision'])
    st.dataframe(styled, use_container_width=True, height=height)


def render_metric_summary(df: pd.DataFrame, target_metric: str = 'far'):
    """Tính FAR hoặc FRR per-variant và hiển thị bảng.

    target_metric: 'far' (impostor data) hoặc 'frr' (owner data).
    """
    if df.empty:
        return

    rows = []
    # Sort theo thứ tự ALL_VARIANTS để bảng nhất quán
    for arch, mode in ALL_VARIANTS:
        key = variant_key(arch, mode)
        sub = df[df['variant'] == key]
        if len(sub) == 0:
            continue

        # Loại bỏ row NO_DATA / ERROR khỏi metric (không có scoring)
        valid = sub[~sub['decision'].isin(['NO_DATA', 'ERROR'])]
        n_total = len(valid)
        n_trusted = (valid['decision'] == 'TRUSTED').sum()
        n_rejected = (valid['decision'] == 'REJECTED').sum()

        if target_metric == 'far':
            rate = n_trusted / n_total if n_total > 0 else 0.0
            rate_name = 'FAR (%)'
        else:
            rate = n_rejected / n_total if n_total > 0 else 0.0
            rate_name = 'FRR (%)'

        thr_used = sub['threshold'].iloc[0] if 'threshold' in sub.columns and len(sub) > 0 else None
        fusion_w_val = sub['fusion_w'].iloc[0] if 'fusion_w' in sub.columns and len(sub) > 0 else None

        rows.append({
            'Model':       ARCH_LABEL[arch],
            'Train mode':  MODE_LABEL[mode],
            'Sessions':    n_total,
            'TRUSTED':     int(n_trusted),
            'REJECTED':    int(n_rejected),
            rate_name:     f"{rate * 100:.2f}",
            'Threshold':   f"{thr_used:.3f}" if thr_used is not None else "—",
            'fusion_w':    f"{fusion_w_val:.2f}" if fusion_w_val is not None else "—",
        })

    if not rows:
        return

    metrics_df = pd.DataFrame(rows)

    # Highlight best/worst FAR or FRR
    rate_col = 'FAR (%)' if target_metric == 'far' else 'FRR (%)'
    vals = metrics_df[rate_col].astype(float)
    best_idx = vals.idxmin()
    worst_idx = vals.idxmax()

    def highlight_row(row):
        styles = [''] * len(row)
        if row.name == best_idx:
            styles = ['background-color: #d4edda;' for _ in row]
        elif row.name == worst_idx and best_idx != worst_idx:
            styles = ['background-color: #f8d7da;' for _ in row]
        return styles

    styled = metrics_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_distribution_subplots(df: pd.DataFrame, title_prefix: str = "Score distribution"):
    """6 subplot histogram, mỗi cái cho 1 variant."""
    if df.empty or 'fused' not in df.columns:
        return

    fig = make_subplots(
        rows=len(ARCHITECTURES), cols=len(DATA_MODES),
        subplot_titles=[
            f"{ARCH_LABEL[a]} × {MODE_LABEL[m]}"
            for a in ARCHITECTURES for m in DATA_MODES
        ],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    for r, arch in enumerate(ARCHITECTURES, start=1):
        for c, mode in enumerate(DATA_MODES, start=1):
            key = variant_key(arch, mode)
            sub = df[df['variant'] == key]
            if len(sub) == 0 or sub['fused'].notna().sum() == 0:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub['fused'].dropna(),
                    nbinsx=15,
                    marker_color='#4C72B0',
                    showlegend=False,
                ),
                row=r, col=c,
            )
            thr = get_threshold_for(key)
            fig.add_vline(
                x=thr, line_dash="dash", line_color="red",
                annotation_text=f"thr={thr:.2f}",
                annotation_position="top right",
                row=r, col=c,
            )

    fig.update_layout(
        title=title_prefix,
        height=200 * len(ARCHITECTURES),
        showlegend=False,
        margin=dict(t=80, b=40, l=40, r=40),
    )
    fig.update_xaxes(range=[0, 1], title_text="P(owner)")
    fig.update_yaxes(title_text="# sessions")
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
    st.subheader(f"Test trên dữ liệu KHÁC của chính {owner_id_active}")

    owner_dir_for_test = Path(st.session_state.get('owner_dir', str(data_dir)))

    # Sessions union của 2 mode để variant nào có data đều test được
    sess_walking = load_user_inertial(owner_id_active, owner_dir_for_test, mode='walking')
    sess_all     = load_user_inertial(owner_id_active, owner_dir_for_test, mode='all')
    common_keys = sorted(set(sess_walking.keys()) | set(sess_all.keys()))

    enroll_keys = next(iter(enrollments.values())).enroll_sessions
    test_sessions = [s for s in common_keys if s not in enroll_keys]

    if not test_sessions:
        st.warning(f"{owner_id_active} đã dùng hết sessions cho enrollment. "
                  f"Giảm 'Số session để enroll' trong sidebar.")
    else:
        st.write(f"Enrollment: `{', '.join(enroll_keys)}` (N={len(enroll_keys)})")
        st.write(f"Test on:    `{', '.join(test_sessions)}` (N={len(test_sessions)})")

        if st.button("▶ Run own-data verification", key="run_own", type="primary"):
            with st.spinner(f"Verifying {len(test_sessions)} sessions × 6 variants..."):
                df = verify_user_sessions_all_variants(
                    owner_id_active, test_sessions, owner_dir_for_test,
                )
                st.session_state['last_own_results'] = df

        if 'last_own_results' in st.session_state:
            df = apply_thresholds_multi(st.session_state['last_own_results'])

            st.markdown("##### FRR per variant (False Reject Rate)")
            render_metric_summary(df, target_metric='frr')

            st.markdown("##### Chi tiết từng session × variant")
            render_results_table(df)


# ─── TAB 2: Single impostor ───────────────────────────────────────
with tab2:
    st.subheader("Test 1 user khác (impostor) — phải bị REJECTED")

    other_users = [u for u in users if u != owner_id_active]
    impostor_id = st.selectbox("Pick impostor user", other_users, key="imp_sel")

    sess_w = load_user_inertial(impostor_id, data_dir, mode='walking')
    sess_a = load_user_inertial(impostor_id, data_dir, mode='all')
    imp_sessions = sorted(set(sess_w.keys()) | set(sess_a.keys()))

    n_imp = st.slider(
        "Số session of impostor để test",
        1, min(4, len(imp_sessions)), min(2, len(imp_sessions)),
        key="imp_n",
    )

    test_imp_sessions = imp_sessions[:n_imp]

    if st.button("▶ Run impostor verification", key="run_imp", type="primary"):
        with st.spinner(f"Verifying {len(test_imp_sessions)} sessions × 6 variants..."):
            df = verify_user_sessions_all_variants(
                impostor_id, test_imp_sessions, data_dir,
            )
            st.session_state['last_imp_results'] = df

    if 'last_imp_results' in st.session_state:
        df = apply_thresholds_multi(st.session_state['last_imp_results'])

        st.markdown("##### FAR per variant (False Accept Rate)")
        render_metric_summary(df, target_metric='far')

        st.markdown("##### Chi tiết từng session × variant")
        render_results_table(df)


# ─── TAB 3: Batch on all impostors ────────────────────────────────
with tab3:
    cohort_impostors = [u for u in users if u != owner_id_active]
    n_cohort_impostors = len(cohort_impostors)

    st.subheader(f"Test trên TẤT CẢ {n_cohort_impostors} users khác — tính FAR thực tế")

    # Max sessions/user = số session ít nhất trong các cohort user
    max_sessions_avail = 1
    if cohort_impostors:
        per_user_counts = []
        for u in cohort_impostors:
            try:
                sw = load_user_inertial(u, data_dir, mode='walking')
                sa = load_user_inertial(u, data_dir, mode='all')
                per_user_counts.append(len(set(sw.keys()) | set(sa.keys())))
            except Exception:
                pass
        if per_user_counts:
            max_sessions_avail = max(1, min(per_user_counts))

    n_per_user = st.slider(
        "Số session/user khi test",
        min_value=1,
        max_value=max_sessions_avail,
        value=min(2, max_sessions_avail),
        key="batch_n",
        help=f"Tối đa {max_sessions_avail} (số session ít nhất trong các cohort user)",
    )

    if st.button("▶ Run batch verification", key="run_batch", type="primary"):
        total_calls = n_cohort_impostors * n_per_user * 6
        with st.spinner(f"Verifying {total_calls} session×variant combos..."):
            df = verify_batch_impostors_all_variants(
                cohort_impostors, n_per_user, data_dir,
            )
            st.session_state['last_batch_results'] = df

    if 'last_batch_results' in st.session_state:
        df = apply_thresholds_multi(st.session_state['last_batch_results'])

        st.markdown("##### FAR per variant (in-cohort impostors)")
        render_metric_summary(df, target_metric='far')

        # Hiển thị FRR luôn nếu có own-data từ Tab 1
        own_df = st.session_state.get('last_own_results')
        if own_df is not None:
            own_df = apply_thresholds_multi(own_df)
            st.markdown("##### FRR per variant (từ Tab 1)")
            render_metric_summary(own_df, target_metric='frr')

        st.markdown("##### Score distribution per variant (impostor)")
        render_distribution_subplots(df, title_prefix="Fused score — impostor sessions")

        st.markdown("##### Chi tiết từng session × variant")
        render_results_table(df, height=500)

        # Liệt kê false accepts
        false_accepts = df[df['decision'] == 'TRUSTED']
        if len(false_accepts) > 0:
            st.warning(f"⚠️ {len(false_accepts)} false-accept rows tổng cộng:")
            st.dataframe(
                false_accepts[['model', 'train_mode', 'test_user', 'session',
                               'fused', 'p_inertial', 'p_touch', 'threshold']],
                use_container_width=True,
            )


# ─── TAB 4: Newbie (unseen) users ─────────────────────────────────
with tab4:
    st.subheader("Test trên users CHƯA TỪNG có trong training cohort")
    st.markdown(
        "**Kịch bản generalization** — model phải reject được user lạ, "
        "không chỉ phân biệt được các user đã thấy trong training."
    )

    if not newbie_users:
        st.warning(
            f"Không tìm thấy newbie users tại `{newbie_dir}`. "
            f"Đặt data theo cùng cấu trúc với `processed_data/`."
        )
    else:
        newbie_pool = [u for u in newbie_users if u != owner_id_active]

        if not newbie_pool:
            st.info(
                f"Owner hiện tại (`{owner_id_active}`) là newbie duy nhất "
                f"trong `{newbie_dir}`. Thêm newbie khác để chạy unseen-user test."
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
                # Max sessions/newbie = số session ít nhất trong pool
                per_nb_counts = []
                for nb in newbie_pool:
                    try:
                        sw = load_user_inertial(nb, newbie_dir, mode='walking')
                        sa = load_user_inertial(nb, newbie_dir, mode='all')
                        per_nb_counts.append(len(set(sw.keys()) | set(sa.keys())))
                    except Exception:
                        pass
                max_sess_newbie = max(1, min(per_nb_counts) if per_nb_counts else 1)
                n_sess_newbie = st.slider(
                    "Sessions/newbie",
                    1, max_sess_newbie, min(2, max_sess_newbie),
                    key="newbie_n_sess",
                    help=f"Tối đa {max_sess_newbie}",
                )

            if mode == "Single newbie (chi tiết)":
                picked = st.selectbox("Pick newbie", newbie_pool, key="newbie_pick")
                if st.button("▶ Run newbie test", key="run_newbie_single",
                             type="primary"):
                    with st.spinner(f"Testing {picked} × 6 variants..."):
                        sw = load_user_inertial(picked, newbie_dir, mode='walking')
                        sa = load_user_inertial(picked, newbie_dir, mode='all')
                        keys = sorted(set(sw.keys()) | set(sa.keys()))[:n_sess_newbie]
                        df = verify_user_sessions_all_variants(picked, keys, newbie_dir)
                        st.session_state['last_newbie_results'] = df
            else:
                if st.button(f"▶ Batch test trên TẤT CẢ {len(newbie_pool)} newbies",
                            key="run_newbie_batch", type="primary"):
                    total_calls = len(newbie_pool) * n_sess_newbie * 6
                    with st.spinner(f"Testing {total_calls} session×variant combos..."):
                        df = verify_batch_impostors_all_variants(
                            newbie_pool, n_sess_newbie, newbie_dir,
                        )
                        st.session_state['last_newbie_results'] = df

        if 'last_newbie_results' in st.session_state:
            df = apply_thresholds_multi(st.session_state['last_newbie_results'])

            st.markdown("##### FAR per variant (UNSEEN newbies)")
            render_metric_summary(df, target_metric='far')

            st.markdown("##### Score distribution per variant (newbie)")
            render_distribution_subplots(df, title_prefix="Fused score — newbie sessions")

            st.markdown("##### Chi tiết từng session × variant")
            render_results_table(df, height=420)

            false_accepts = df[df['decision'] == 'TRUSTED']
            if len(false_accepts) > 0:
                st.warning(f"⚠ {len(false_accepts)} false-accept rows trên newbies:")
                st.dataframe(
                    false_accepts[['model', 'train_mode', 'test_user', 'session',
                                   'fused', 'p_inertial', 'p_touch', 'threshold']],
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Methodology: train per-user RF (inertial 128-D embedding + touch 48-D), "
    "tune fusion_w bằng grid-search AUC trên held-out val session. "
    "Adaptive threshold tính tại điểm EER từ val set lúc enroll. "
    "Impostor pool build on-the-fly (no leakage). "
    "Mỗi test session được score qua cả 6 biến thể model để so sánh trực tiếp."
)
